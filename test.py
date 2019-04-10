import numpy as np
import scipy.misc
import scipy.io
import time, sys
import threading
import utils
import tensorflow as tf
import tflearn
import data
import options
sys.path.append('./graph')
sys.path.append('./evaluation_metric')
import graph
from graph_ae import encoder, decoder, discriminator
import os

print(utils.toYellow("======================================================="))
print(utils.toYellow("test.py "))
print(utils.toYellow("======================================================="))
print(utils.toMagenta("setting configurations..."))
opt = options.set()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id 
utils.mkdir(opt.savedir + "/models_{0}".format(opt.group))
stage = opt.stage
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

    ## ------ evaluation metric -----
    pcgt_s1 = tf.tile(pcgt,[opt.rand_num,1,1]) #[R*B*V,N,3]
    pcgt_s2_test = tf.reshape(pcgt,[-1, opt.num_images, opt.pcgtSize, 3])[:,0,:,:] #[B,N,3]
    pc_s1 = pc #[R*B*V,N,3]
    pc_s2 = tf.reshape(pc, [-1, opt.num_images*opt.pcgtSize, 3]) #train: [R*B,N*V,3], test: [B,N*V,3]

    emd_s1 = utils.evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=True)
    cd_s1, d1_s1, d2_s1 = utils.evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=False)
    cd_fps_s2_evaluate, d1_fps_s2_evaluate, d2_fps_s2_evaluate = utils.evaluation(gt=pcgt_s2_test, pred=utils.farthest_point_sampling(opt.pcgtSize, pc_s2), is_emd=False)
    cd_s2_evaluate, d1_s2_evaluate, d2_s2_evaluate = utils.evaluation(gt=pcgt_s2_test, pred=pc_s2, is_emd=False)

    ## ------ evaluation loss ------
    K_tile = tf.tile(K, [opt.rand_num, 1, 1]) #[R*B*V,3,3]
    extrinsic_tile = tf.tile(extrinsic, [opt.rand_num, 1, 1]) #[R*B*V,4,4]
    depth_tile = tf.tile(tf.squeeze(depth, [3]), [opt.rand_num, 1, 1]) #[R*B*V,H,W]
    loss_front_s1 = utils.front_loss(pc, K_tile, extrinsic_tile, depth_tile, H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True)
    pc_concat = tf.reshape(pc, [-1, opt.num_images*opt.pcgtSize,3])
    pc_concat = tf.reshape(tf.tile(tf.expand_dims(pc_concat,axis=1), [1,opt.num_images, 1,1]),[-1, opt.num_images*opt.pcgtSize, 3])
    loss_front_s2_evaluate = utils.front_loss(pc_concat, K, extrinsic, tf.squeeze(depth, [3]), H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True, is_emd=False)

    ## ------ inference loss ------
    extrinsic_consis = tf.tile(tf.expand_dims(tf.eye(4), axis=0), [tf.shape(pc)[0], 1, 1])
    
    consis_loss_all_cd = utils.consis_loss(tf.reshape(pc, [-1, opt.num_images, opt.pcgtSize, 3]), tf.reshape(extrinsic_consis, [-1, opt.num_images, 4, 4]), opt.num_images, is_emd=False)
    consis_loss_cd = tf.reduce_mean(consis_loss_all_cd)
    r_fgsm, loss_fgsm = utils.FGSM(r, consis_loss_cd, opt.fgsm_lr) 


    ## ------ optimizer ------
    lr_PH = tf.placeholder(tf.float32, shape=[])
# ------ load data -------
print(utils.toMagenta("loading dataset..."))

test_loader = data.Loader(opt, loadDepth=True, loadPCGT=True, loadPose=True, category_list=opt.cat_list, data_type='test')


# ------ start session ------
print(utils.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfConfig.allow_soft_placement = True
tfConfig.log_device_placement = False

with tf.Session(config=tfConfig) as sess:
    sess.run(tf.global_variables_initializer())
    print(utils.toMagenta("start evaluation..."))
    if stage == 1:
        runList_test = [loss_front, cd_s1, emd_s1, d1_s1, d2_s1]
        appendix = []
        metric_list = ['cd', 'emd']
    else:
        runList_test = [loss_front_s2_evaluate, cd_s2_evaluate, cd_fps_s2_evaluate, d1_s2_evaluate, d2_s2_evaluate]
        appendix = [r_fgsm,loss_fgsm,consis_loss_all_cd]
        metric_list = ['cd', 'cd_fps']
    if opt.load:
        utils.restoreModel(opt.savedir, opt, sess, tf.train.Saver())
        print(utils.toMagenta("loading pretrained ({0}) to evaluate...".format(opt.load)))
        opt.load = None

    loss_test_all = data.perform_test(opt=opt, sess=sess, PH=PH, runList=runList_test, test_loader=test_loader, stage=stage, appendix=appendix)
    [loss_front_test, test1, test2, d1, d2] = loss_test_all
    print(utils.toYellow("======================================================="))
    print("test loss_front: {0}".format(utils.toRed("{0:.3f}".format(loss_front_test))))
    print("test {0}: {1}".format(metric_list[0], utils.toRed("{0:.3f}".format(test1))))
    print("test {0}: {1}".format(metric_list[1], utils.toRed("{0:.3f}".format(test2))))
    print("test d1: {0}".format(utils.toRed("{0:.3f}".format(d1))))
    print("test d2: {0}".format(utils.toRed("{0:.3f}".format(d2))))
    print(utils.toYellow("======================================================="))




print(utils.toYellow("======= EVALUATION DONE ======="))



