import numpy as np
import scipy.misc
import scipy.io
import time, sys
import threading
import utils
import tensorflow as tf
import tflearn
import data
import saverhelper as helper
import options
sys.path.append('./graph')
sys.path.append('./evaluation_metric')
import graph
from graph_ae import encoder, decoder, discriminator
import os
import pdb

print(utils.toYellow("======================================================="))
print(utils.toYellow("train.py "))
print(utils.toYellow("======================================================="))
print(utils.toMagenta("setting configurations..."))
opt = options.set()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id 
utils.mkdir(opt.savedir + "/models_{0}".format(opt.group))
if opt.cat == 1:
    test_iter = 2000
elif opt.cat == 13:
    test_iter = 5000
else:
    raise('Error!')
print(utils.toMagenta("building graph..."))
tf.reset_default_graph()

# ------ define graph ------
with tf.device("/gpu:0"):
    ## ------ define input data ------
    r = tf.placeholder(tf.float32, shape=[None, opt.rand_dim])
    image_in = tf.placeholder(tf.float32, shape=[None, opt.in_H, opt.in_W, 3])
    pcgt = tf.placeholder(tf.float32, shape=[None, opt.pcgtSize, 3])
    depth = tf.placeholder(tf.float32, shape=[None, opt.depth_H, opt.depth_W, 1])
    mask = tf.placeholder(tf.float32, shape=[None, opt.depth_H, opt.depth_W, 1])
    K = tf.placeholder(tf.float32, shape=[None, 3, 3])
    extrinsic = tf.placeholder(tf.float32, shape=[None, 4, 4])
    PH = [image_in, pcgt, depth, mask, K, extrinsic, r]

    ## ------ build single-image encoder model ------
    tflearn.init_graph(seed=1029, num_cores=2, gpu_memory_fraction=0.9, soft_placement=True)
    pc  = graph.encoder(opt, image_in, r)
    z_syn = encoder(pc)
    z_real = encoder(tf.tile(pcgt,[opt.rand_num,1,1]),reuse=True)
    prob_syn, logit_syn = discriminator(z_syn)
    prob_real, logit_real = discriminator(z_real, reuse=True)
    train_vars = tf.trainable_variables()
    d_vars = [v for v in train_vars if 'discriminator' in v.name]
    g_vars = [v for v in train_vars if ('discriminator'  not in v.name and 'encoder' not in v.name)]
    e_vars = [v for v in train_vars if 'encoder' in v.name]

    ## ------ evaluation metric -----
    pcgt_s1 = tf.tile(pcgt,[opt.rand_num,1,1]) #[R*B*V,N,3]
    pcgt_s2_train = tf.reshape(tf.tile(pcgt,[opt.rand_num,1,1]),[-1, opt.num_images, opt.pcgtSize, 3])[:,0,:,:] #[R*B,N,3]
    pcgt_s2_test = tf.reshape(pcgt,[-1, opt.num_images, opt.pcgtSize, 3])[:,0,:,:] #[B,N,3]
    pc_s1 = pc #[R*B*V,N,3]
    pc_s2 = tf.reshape(pc, [-1, opt.num_images*opt.pcgtSize, 3]) #train: [R*B,N*V,3], test: [B,N*V,3]

    emd_s1 = utils.evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=True)
    cd_s1, d1_s1, d2_s1 = utils.evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=False)
    cd_fps_s2, d1_fps_s2, d2_fps_s2 = utils.evaluation(gt=pcgt_s2_train, pred=utils.farthest_point_sampling(opt.pcgtSize, pc_s2), is_emd=False)
    cd_s2, d1_s2, d2_s2 = utils.evaluation(gt=pcgt_s2_train, pred=pc_s2, is_emd=False)
    cd_fps_s2_evaluate, d1_fps_s2_evaluate, d2_fps_s2_evaluate = utils.evaluation(gt=pcgt_s2_test, pred=utils.farthest_point_sampling(opt.pcgtSize, pc_s2), is_emd=False)
    cd_s2_evaluate, d1_s2_evaluate, d2_s2_evaluate = utils.evaluation(gt=pcgt_s2_test, pred=pc_s2, is_emd=False)

    ## ------ train loss ------
    ### ------ diversity constraint ------
    r_reshape = tf.reshape(r, shape=[opt.rand_num, -1, opt.rand_dim]) #[R,B*V,D]
    pc_reshape = tf.reshape(pc,shape=[opt.rand_num, -1, opt.pcgtSize, 3]) #[R,B*V,N,3]
    loss_div_s1 = utils.get_loss_diversity(r_reshape, pc_reshape, opt.rand_num, alpha=opt.alpha_s1, is_emd=True)
    loss_div_s2 = utils.get_loss_diversity(r_reshape, pc_reshape, opt.rand_num, alpha=opt.alpha_s2, is_emd=True)

    ### ------ front constraint ------
    K_tile = tf.tile(K, [opt.rand_num, 1, 1]) #[R*B*V,3,3]
    extrinsic_tile = tf.tile(extrinsic, [opt.rand_num, 1, 1]) #[R*B*V,4,4]
    depth_tile = tf.tile(tf.squeeze(depth, [3]), [opt.rand_num, 1, 1]) #[R*B*V,H,W]
    loss_front_s1 = utils.front_loss(pc, K_tile, extrinsic_tile, depth_tile, H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True)
    pc_concat = tf.reshape(pc, [-1, opt.num_images*opt.pcgtSize,3])
    pc_concat = tf.reshape(tf.tile(tf.expand_dims(pc_concat,axis=1), [1,opt.num_images, 1,1]),[-1, opt.num_images*opt.pcgtSize, 3])
    loss_front_s2 = utils.front_loss(pc_concat, K_tile, extrinsic_tile, depth_tile,  H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True, is_emd=False)

    ### ------ adversial loss ------
    d_loss = tf.reduce_mean(logit_syn) - tf.reduce_mean(logit_real)
    g_loss = -tf.reduce_mean(logit_syn)

    ndims = z_real.get_shape().ndims
    batch_size = tf.shape(z_real)[0]
    alpha = tf.random_uniform(shape=[batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
    differences = z_syn - z_real
    interpolates = z_real + (alpha * differences)
    gradients = tf.gradients(discriminator(interpolates, reuse=True)[1], [interpolates])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, ndims)))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    d_loss += 10 * gradient_penalty


    loss_s1 = loss_front_s1 + opt.loss_weight_div_s1*loss_div_s1  + opt.loss_weight_adv*g_loss
    loss_s2 = loss_front_s2 + opt.loss_weight_div_s2*loss_div_s2  + opt.loss_weight_adv*g_loss
    loss_d = opt.loss_weight_adv*d_loss

    ## ------ evaluation loss (only for stage2) ------
    loss_front_s2_evaluate = utils.front_loss(pc_concat, K, extrinsic, tf.squeeze(depth, [3]), H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True, is_emd=False)

    ## ------ consistency loss (only for inference) ------
    extrinsic_consis = tf.tile(tf.expand_dims(tf.eye(4), axis=0), [tf.shape(pc)[0], 1, 1])
    consis_loss_all_cd = utils.consis_loss(tf.reshape(pc, [-1, opt.num_images, opt.pcgtSize, 3]), tf.reshape(extrinsic_consis, [-1, opt.num_images, 4, 4]), opt.num_images, is_emd=False)
    consis_loss_cd = tf.reduce_mean(consis_loss_all_cd)
    r_fgsm, loss_fgsm = utils.FGSM(r, consis_loss_cd, opt.fgsm_lr) 


    ## ------ optimizer ------
    lr_PH = tf.placeholder(tf.float32, shape=[])
    optim_s1 = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss_s1, var_list=g_vars)
    optim_s2 = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss_s2, var_list=g_vars)
    optim_d = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss_d, var_list=d_vars)


# ------ load data ------
print(utils.toMagenta("loading dataset..."))

train_loader = data.Loader(opt, loadDepth=True, loadPCGT=True, loadPose=True, category_list=opt.cat_list, data_type='train')
test_loader = data.Loader(opt, loadDepth=True, loadPCGT=True, loadPose=True, category_list=opt.cat_list, data_type='test')

train_loader.loadChunk(opt)

# ------ start session ------
print(utils.toYellow("======= TRAINING START ======="))
timeStart = time.time()
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfConfig.allow_soft_placement = True
tfConfig.log_device_placement = False
with tf.Session(config=tfConfig) as sess:
    sess.run(tf.global_variables_initializer())
    model_name = opt.model
    print(utils.toMagenta("start training..."))
    saver = tf.train.Saver(e_vars)
    if opt.cat == 1 :
        saver.restore(sess,'output/models_ae/ae_cat1.ckpt')
    elif opt.cat == 13:
        saver.restore(sess,'output/models_ae/ae_cat13.ckpt')
    else:
        raise('Error!')

    for stage in range(1,3):
        itPerChunk = opt.chunkSize // opt.batchSize
        if stage == 1:
            saverhelper_s1 = helper.SaverHelper(opt, max_num=1, stage=1)
            chunkResumeN, chunkMaxN = opt.fromIt_stage1//itPerChunk, opt.toIt_stage1//itPerChunk
            runList_train = [optim_s1, loss_s1, cd_s1, emd_s1, loss_front_s1, loss_div_s1, g_loss]
            runList_test = [loss_front_s1, cd_s1, emd_s1, d1_s1, d2_s1]
            appendix = []
            saverhelper = saverhelper_s1
            metric_list = ['cd', 'emd']
        else:
            saverhelper.restoreModelFromBest(sess)
            saverhelper_s2 = helper.SaverHelper(opt, max_num=1, stage=2)
            chunkResumeN, chunkMaxN = opt.fromIt_stage2//itPerChunk, opt.toIt_stage2//itPerChunk
            runList_train = [optim_s2, loss_s2, cd_s2, cd_fps_s2, loss_front_s2, loss_div_s2, g_loss]
            runList_test = [loss_front_s2_evaluate, cd_s2_evaluate, cd_fps_s2_evaluate, d1_fps_s2_evaluate, d2_fps_s2_evaluate]
            appendix = [r_fgsm,loss_fgsm,consis_loss_all_cd]
            saverhelper = saverhelper_s2
            metric_list = ['cd', 'cd_fps']
        if opt.load:
            saverhelper.restoreModel(opt.savedir, opt, sess)
            print(utils.toMagenta("loading pretrained ({0}) to fine-tune...".format(opt.load)))
            opt.load = None

        opt.model = model_name + '_s' + str(stage)
        for c in range(chunkResumeN, chunkMaxN):
            train_loader.shipChunk()
            train_loader.thread = threading.Thread(target=train_loader.loadChunk, args=[opt])
            train_loader.thread.start()
            for i in range(c*itPerChunk, (c+1)*itPerChunk):
                lr = opt.lr*opt.lrDecay**(i//opt.lrStep)
                batch = data.makeBatch(opt, train_loader, PH, [(i-c*itPerChunk)*opt.batchSize, (i+1-c*itPerChunk)*opt.batchSize], is_rand=True)
                batch[lr_PH] = lr
                batch[r] = np.reshape(batch[r], (-1,opt.rand_dim))
                batch[image_in] = np.tile(batch[image_in], (opt.rand_num, 1, 1, 1))
                _, d_loss_ = sess.run([optim_d, d_loss], feed_dict=batch)
                _, loss_, train1, train2, loss_front_, loss_div_, g_loss_  = sess.run(runList_train, feed_dict=batch)
                if (i+1)%20 == 0:
                    print("stage{9} : it. {0}/{1}, lr={2}, {10}={4}, {11}={5}, loss={6}, loss_front={7}, loss_div={8}, loss_g={12}, loss_d={13}, time={3}"
                        .format(utils.toCyan("{0}".format(i+1)),
                                chunkMaxN*itPerChunk,
                                utils.toYellow("{0:.0e}".format(lr)),
                                utils.toGreen("{0:.3f}".format(time.time()-timeStart)),
                                utils.toRed("{0:.3f}".format(train1)),
                                utils.toRed("{0:.3f}".format(train2)),
                                utils.toRed("{0:.3f}".format(loss_)),
                                utils.toRed("{0:.3f}".format(loss_front_)), 
                                utils.toRed("{0:.3f}".format(loss_div_)),
                                stage,
                                metric_list[0],
                                metric_list[1],
                                utils.toRed("{0:.3f}".format(g_loss_)),
                                utils.toRed("{0:.3f}".format(d_loss_))))
                ## ------ evaluation ------
                if (i+1)%test_iter == 0:
                    loss_test_all = data.perform_test(opt=opt, sess=sess, PH=PH, runList=runList_test, test_loader=test_loader, stage=stage, appendix=appendix)
                    [loss_front_test, test1, test2, d1, d2] = loss_test_all
                    saverhelper.saveModel(opt.savedir, opt, sess, i+1, [train2, train1, test2, test1, d1, d2])
                    print(utils.toGreen("model saved: {0}/{1}_it{2}".format(opt.group, opt.model, i+1)))
                   
                    cd_best = saverhelper.report_best()
                    print(utils.toYellow("======================================================="))
                    print("test loss_front: {0}".format(utils.toRed("{0:.3f}".format(loss_front_test))))
                    print("test {0}: {1}".format(metric_list[0], utils.toRed("{0:.3f}".format(test1))))
                    print("test {0}: {1}".format(metric_list[1], utils.toRed("{0:.3f}".format(test2))))
                    print("test d1: {0}".format(utils.toRed("{0:.3f}".format(d1))))
                    print("test d2: {0}".format(utils.toRed("{0:.3f}".format(d2))))
                    print("best-to-now cd: {0}".format(utils.toRed("{0:.3f}".format(cd_best))))
                    print(utils.toYellow("======================================================="))


            train_loader.thread.join()


print(utils.toYellow("======= TRAINING DONE ======="))

