import numpy as np
import utils
import os
import tensorflow as tf
import heapq
import time
import string

class SaverHelper():

    # ------ initialization ------
    def __init__(self, opt, max_num=5, stage=1):
        self.saver = tf.train.Saver(max_to_keep=None)
        self.stage = stage
        self.max_num = max_num 
        self.container = []
        self.cd_fps_timeline = []
        self.emd_timeline = []
        self.cd_timeline = []
        self.model_timeline = []
        now = string.split(time.ctime())
        fname = now[1] + '_' + now[2] + '_' + now[3]
        self.filepath = os.path.join(opt.diary_path, fname + '.txt')
        with open(self.filepath, 'w') as file:
            file.write(opt.task + '\n')

    def restoreModelFromIt(self, path, opt, sess, it):
        utils.restoreModelFromIt(path, opt, sess, self.saver, it)

    def restoreModel(self, path, opt, sess):
        utils.restoreModel(path, opt, sess, self.saver)

    def restoreModelFromBest(self, sess):
        best_model = self.report_best_model()
        self.saver.restore(sess, best_model)        
        print("Loading best model from stage1: "+best_model)

    # ------ preserve best max_num models ------
    def saveModel(self, path, opt, sess, it, info):
        utils.saveModel(path, opt, sess, self.saver, it)
        savedir = './' + path + "/models_{0}/{1}_it{2}.ckpt".format(opt.group, opt.model, it)
        
        if self.stage == 2:
            cd_fps_loss = info[0]
            cd_loss = info[1]
            cd_fps_test = info[2]
            cd_test = info[3]
            d1 = info[4]
            d2 = info[5]
            self.cd_fps_timeline.append(cd_fps_test)
            self.cd_timeline.append(cd_test)
            self.model_timeline.append(savedir)
            info_ = (-cd_fps_test, savedir)
        else:
            emd_loss = info[0]
            cd_loss = info[1]
            emd_test = info[2]
            cd_test = info[3]
            d1 = info[4]
            d2 = info[5]
            self.emd_timeline.append(emd_test)
            self.cd_timeline.append(cd_test)
            self.model_timeline.append(savedir)
            info_ = (-cd_test, savedir)
  
        ## ------ remove worst model ------
        heapq.heappush(self.container, info_)
        if len(self.container) > self.max_num:
            to_delete = self.container[0]
            self.removeModel(to_delete[1])
            self.container = self.container[1:self.max_num+1] 

        ## ------ update diary ------
        if self.stage == 2:
            with open(self.filepath, 'a') as file:
                file.write("SAVEDIR:{0},CD_fps_train:{1},CD_train:{2},CD_fps_test:{3},CD_test:{4},d1:{5},d2:{6}\n".format(savedir, cd_fps_loss, cd_loss, cd_fps_test, cd_test,d1,d2))
        else:
            with open(self.filepath, 'a') as file:
                file.write("SAVEDIR:{0},EMD_train:{1},CD_train:{2},EMD_test:{3},CD_test:{4},d1:{5},d2:{6}\n".format(savedir, emd_loss, cd_loss, emd_test, cd_test, d1, d2))


    def removeModel(self, path):
        os.remove(path + '.data-00000-of-00001')
        os.remove(path + '.meta')
        os.remove(path + '.index')

    def report_best(self):
        if self.stage == 1:
            return min(self.cd_timeline)
        else:
            return min(self.cd_fps_timeline)

    def report_best_model(self):
        if self.stage == 1:
            return self.model_timeline[np.argmin(self.cd_timeline)]
        else:
            return self.model_timeline[np.argmin(self.cd_fps_timeline)]

