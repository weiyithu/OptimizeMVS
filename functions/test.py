
import numpy as np
import time, sys
import threading
import tensorflow as tf

this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_dir)
from build_graph import build_test_graph
sys.path.append(os.path.join(this_dir, '..'))
from config.base_cfg import init_config
import lib.diary.saverhelper as helper
from lib.loader.dataloader import Loader
import lib.utils.color_utils as color_utils
from lib.utils.color_utils import *

print(toYellow("======================================================="))
print(toYellow("test.py "))

# ------ setting configs ------
print(toYellow("======================================================="))
print(toMagenta("setting configurations..."))

args = init_config()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
stage = args.stage

build_test_graph(args)
# ------ load data -------
print(toMagenta("loading dataset..."))
test_loader = data.Loader(args, loadDepth=True, loadPCGT=True, loadPose=True, category_list=args.cat_list, data_type='test')

# ------ start session ------
print(toYellow("======= EVALUATION START ======="))
timeStart = time.time()
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfConfig.allow_soft_placement = True
tfConfig.log_device_placement = False

with tf.Session(config=tfConfig) as sess:
    sess.run(tf.global_variables_initializer())
    print(toMagenta("start evaluation..."))
    if stage == 1:
        runList_test = [loss_front, cd_s1, emd_s1, d1_s1, d2_s1]
        appendix = []
        metric_list = ['cd', 'emd']
    else:
        runList_test = [loss_front_s2_evaluate, cd_s2_evaluate, cd_fps_s2_evaluate, d1_s2_evaluate, d2_s2_evaluate]
        appendix = [r_fgsm,loss_fgsm,consis_loss_all_cd]
        metric_list = ['cd', 'cd_fps']
    if args.load:
        utils.restoreModel(args.savedir, args, sess, tf.train.Saver())
        print(toMagenta("loading pretrained ({0}) to evaluate...".format(args.load)))
        args.load = None

    loss_test_all = data.perform_test(opt=args, sess=sess, PH=PH, runList=runList_test, test_loader=test_loader, stage=stage, appendix=appendix)
    [loss_front_test, test1, test2, d1, d2] = loss_test_all
    print(toYellow("======================================================="))
    print("test loss_front: {0}".format(toRed("{0:.3f}".format(loss_front_test))))
    print("test {0}: {1}".format(metric_list[0], toRed("{0:.3f}".format(test1))))
    print("test {0}: {1}".format(metric_list[1], toRed("{0:.3f}".format(test2))))
    print("test d1: {0}".format(toRed("{0:.3f}".format(d1))))
    print("test d2: {0}".format(toRed("{0:.3f}".format(d2))))
    print(toYellow("======================================================="))

print(toYellow("======= EVALUATION DONE ======="))

