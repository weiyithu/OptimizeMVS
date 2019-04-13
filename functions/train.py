
import numpy as np
import os, sys, time
import threading
import tensorflow as tf

this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_dir)
from build_graph import build_train_graph
sys.path.append(os.path.join(this_dir, '..'))
from config.base_cfg import init_config
import lib.diary.saverhelper as helper
from lib.loader.dataloader import Loader, makeBatch, perform_test
from lib.utils.color_utils import * 
from lib.utils.io_utils import mkdir

print(toYellow("======================================================="))
print(toYellow("train.py "))

# ------ setting configs ------
print(toYellow("======================================================="))
print(toMagenta("setting configurations..."))

args = init_config()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
BASEDIR = os.path.join(this_dir, '..')
mkdir(os.path.join(BASEDIR, args.savedir))
mkdir(os.path.join(BASEDIR, os.path.join(args.savedir, "models_{0}".format(args.group))))
mkdir(os.path.join(BASEDIR, args.diary_path))

tlist = build_train_graph(args)
# ------ load data ------
print(toMagenta("loading dataset..."))

train_loader = Loader(args, loadDepth=True, loadPCGT=True, loadPose=True, category_list=args.cat_list, data_type='train')
test_loader = Loader(args, loadDepth=True, loadPCGT=True, loadPose=True, category_list=args.cat_list, data_type='test')

train_loader.loadChunk(args)

# ------ start session ------
print(toYellow("======= TRAINING START ======="))
timeStart = time.time()
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = False
tfConfig.allow_soft_placement = True
tfConfig.log_device_placement = False
with tf.Session(config=tfConfig) as sess:
    sess.run(tf.global_variables_initializer())
    model_name = args.model
    print(toMagenta("start training..."))
    saver = tf.train.Saver(tlist['e_vars'])
    if args.cat == 1 :
        saver.restore(sess,'pretrain/models_ae/ae_cat1.ckpt')
    elif args.cat == 13:
        saver.restore(sess,'pretrain/models_ae/ae_cat13.ckpt')

    for stage in range(1,3):
        itPerChunk = args.chunkSize // args.batchSize
        if stage == 1:
            saverhelper_s1 = helper.SaverHelper(args, max_num=1, stage=1)
            chunkResumeN, chunkMaxN = args.fromIt_stage1//itPerChunk, args.toIt_stage1//itPerChunk
            runList_train = [tlist['optim_s1'], tlist['loss_s1'], tlist['cd_s1'], tlist['emd_s1'], tlist['loss_front_s1'], tlist['loss_div_s1'], tlist['g_loss']]
            runList_test = [tlist['loss_front_s1'], tlist['cd_s1'], tlist['emd_s1'], tlist['d1_s1'], tlist['d2_s1']]
            appendix = []
            saverhelper = saverhelper_s1
            metric_list = ['cd', 'emd']
        else:
            saverhelper.restoreModelFromBest(sess)
            saverhelper_s2 = helper.SaverHelper(args, max_num=1, stage=2)
            chunkResumeN, chunkMaxN = args.fromIt_stage2//itPerChunk, args.toIt_stage2//itPerChunk
            runList_train = [tlist['optim_s2'], tlist['loss_s2'], tlist['cd_s2'], tlist['cd_fps_s2'], tlist['loss_front_s2'], tlist['loss_div_s2'], tlist['g_loss']]
            runList_test = [tlist['loss_front_s2_evaluate'], tlist['cd_s2_evaluate'], tlist['cd_fps_s2_evaluate'], tlist['d1_fps_s2_evaluate'], tlist['d2_fps_s2_evaluate']]
            appendix = [tlist['r_fgsm'], tlist['loss_fgsm'], tlist['consis_loss_all_cd']]
            saverhelper = saverhelper_s2
            metric_list = ['cd', 'cd_fps']
        if args.load:
            saverhelper.restoreModel(args.savedir, args, sess)
            print(toMagenta("loading pretrained ({0}) to fine-tune...".format(args.load)))
            args.load = None

        args.model = model_name + '_s' + str(stage)
        for c in range(chunkResumeN, chunkMaxN):
            train_loader.shipChunk()
            train_loader.thread = threading.Thread(target=train_loader.loadChunk, args=[args])
            train_loader.thread.start()
            for i in range(c * itPerChunk, (c+1) * itPerChunk):
                lr = args.lr * args.lrDecay ** (i // args.lrStep)
                batch = makeBatch(args, train_loader, tlist['PH'], [(i-c*itPerChunk)*args.batchSize, (i+1-c*itPerChunk)*args.batchSize], is_rand=True)
                batch[tlist['lr_PH']] = lr
                batch[tlist['r']] = np.reshape(batch[tlist['r']], (-1, args.rand_dim))
                batch[tlist['image_in']] = np.tile(batch[tlist['image_in']], (args.rand_num, 1, 1, 1))
                _, d_loss_ = sess.run([tlist['optim_d'], tlist['d_loss']], feed_dict=batch)
                _, loss_, train1, train2, loss_front_, loss_div_, g_loss_  = sess.run(runList_train, feed_dict=batch)
                if (i+1)%20 == 0:
                    print("stage{9} : it. {0}/{1}, lr={2}, {10}={4}, {11}={5}, loss={6}, loss_front={7}, loss_div={8}, loss_g={12}, loss_d={13}, time={3}"
                        .format(toCyan("{0}".format(i+1)),
                                chunkMaxN*itPerChunk,
                                toYellow("{0:.0e}".format(lr)),
                                toGreen("{0:.3f}".format(time.time()-timeStart)),
                                toRed("{0:.3f}".format(train1)),
                                toRed("{0:.3f}".format(train2)),
                                toRed("{0:.3f}".format(loss_)),
                                toRed("{0:.3f}".format(loss_front_)),
                                toRed("{0:.3f}".format(loss_div_)),
                                stage,
                                metric_list[0],
                                metric_list[1],
                                toRed("{0:.3f}".format(g_loss_)),
                                toRed("{0:.3f}".format(d_loss_))))
                ## ------ evaluation ------
                if (i+1) % args.test_iter == 0:
                    loss_test_all = perform_test(args=args, sess=sess, PH=tlist['PH'], runList=runList_test, test_loader=test_loader, stage=stage, appendix=appendix)
                    [loss_front_test, test1, test2, d1, d2] = loss_test_all
                    saverhelper.saveModel(args.savedir, args, sess, i+1, [train2, train1, test2, test1, d1, d2])
                    print(toGreen("model saved: {0}/{1}_it{2}".format(args.group, args.model, i+1)))
                   
                    cd_best = saverhelper.report_best()
                    print(toYellow("======================================================="))
                    print("test loss_front: {0}".format(toRed("{0:.3f}".format(loss_front_test))))
                    print("test {0}: {1}".format(metric_list[0], toRed("{0:.3f}".format(test1))))
                    print("test {0}: {1}".format(metric_list[1], toRed("{0:.3f}".format(test2))))
                    print("test d1: {0}".format(toRed("{0:.3f}".format(d1))))
                    print("test d2: {0}".format(toRed("{0:.3f}".format(d2))))
                    print("best-to-now cd: {0}".format(toRed("{0:.3f}".format(cd_best))))
                    print(toYellow("======================================================="))


            train_loader.thread.join()


print(toYellow("======= TRAINING DONE ======="))

