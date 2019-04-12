import numpy as np
import threading
import scipy.io
import os, sys
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_dir, '..', 'utils'))
from io_utils import read_from_list, imread, imread_depth, read_pc_from_ply, read_pc_from_pcd
from color_utils import *

class Loader():
    # ------ initialization ------
    def __init__(self, args, loadDepth=True, loadPose=True, 
                        loadPCGT=True, category_list=None, data_type=None):

        self.datatype = data_type

        self.loadDepth = loadDepth
        self.loadPose = loadPose
        self.loadPCGT = loadPCGT
        self.num_images = args.num_images

        ## ------ get data list ------
        self.data = []
        if self.datatype == 'train':
            base_path = 'train'
        elif self.datatype == 'test':
            base_path = 'test' 
        else:
            base_path = 'all'

        if category_list is None:
            self.data_list = read_from_list(os.path.join(args.datalist_path, base_path + '.list'))
        else:
            self.data_list = []
            for cat in category_list:
                objs = read_from_list(os.path.join(args.datalist_path, base_path, cat + '.list'))
                self.data_list.extend(objs)

        ## ------ handle corner case ------
        corner_case_list = ['03624134/67ada28ebc79cc75a056f196c127ed77', '04074963/b65b590a565fa2547e1c85c5c15da7fb', '04090263/4a32519f44dc84aabafe26e2eb69ebf4']
        if args.pcgtSize == 2048:
            for name in corner_case_list:
                if name in self.data_list:
                    self.data_list.remove(name)

    # ------ get length of the data list ------
    def length(self):
        return len(self.data_list)

    # ------ load Chunk ------
    def loadChunk(self, args, loadRange=None):
        data = {}
        if loadRange is not None: idx = np.arange(loadRange[0],loadRange[1])
        else: idx = np.random.permutation(len(self.data_list))[:args.chunkSize]
        chunkSize = len(idx)

        ## ------ preallocate memory ------
        data['image_in'] = np.ones([chunkSize, self.num_images, args.in_H, args.in_W, 3], dtype=np.float32)
        if self.loadDepth:
            data['depth'] = np.ones([chunkSize, self.num_images, args.depth_H, args.depth_W], dtype=np.float32)
            data['mask'] = np.ones([chunkSize, self.num_images, args.depth_H, args.depth_W], dtype=np.bool)
        if self.loadPCGT:
            data['pcgt'] = np.ones([chunkSize, args.pcgtSize, 3], dtype=np.float32)
        if self.loadPose:
            data['extrinsic'] = np.ones([chunkSize, self.num_images, 4, 4])

        ## ------ load data ------
        for c in range(chunkSize):
            obj = self.data_list[idx[c]]

            ### ------ load rgb ------
            obj_path = os.path.join(args.rendering_path, obj)
            rgb_data = []
            for i in range(self.num_images):
                suffix = str(i) + '.png'
                im = imread(os.path.join(obj_path, 'render_' + suffix))
                rgb_data.append(im)
            data["image_in"][c] = np.asarray(rgb_data)

            ### ------ load depth ------
            if self.loadDepth:
                obj_path = os.path.join(args.rendering_path, obj)
                depth_data = []
                mask_data = []
                for i in range(self.num_images):
                    suffix = str(i) + '.png'
                    depth = imread_depth(os.path.join(obj_path, 'depth_' + suffix))
                    depth_data.append(depth)

                    mask = depth != 65535
                    mask_data.append(mask)

                data["depth"][c] = np.asarray(depth_data)
                data["mask"][c] = np.asarray(mask_data)

            if args.data_use_mask == True:
                data["image_in"][c] = data["image_in"][c] - 1.0
                mask_temp = data["mask"][c][:,:,:,np.newaxis].copy()
                mask_temp = np.tile(mask_temp, (1,1,1,3))
                data["image_in"][c] = data["image_in"][c]*mask_temp
                data["image_in"][c] = data["image_in"][c] + 1.0

            ### ------ load point cloud groundtruth ------
            if self.loadPCGT:
                obj_path = os.path.join(args.pcgt_path, obj)
                if args.pcgtSize == 2048:
                    data['pcgt'][c] = read_pc_from_ply(obj_path + '.ply')
                else:
                    data['pcgt'][c] = read_pc_from_pcd(obj_path + '.pcd')

            ### ------ load extrinsic data ------
            if self.loadPose:
                obj_path = os.path.join(args.rendering_path, obj)
                extrinsic_data = []
                for i in range(self.num_images):
                    suffix = str(i) + '.mat'
                    extrinsic_data.append(scipy.io.loadmat(os.path.join(obj_path, 'camera_' + suffix))['extrinsic'])
                data['extrinsic'][c] = np.asarray(extrinsic_data)

        self.pendingChunk = data

    def shipChunk(self):
        self.readyChunk,self.pendingChunk = self.pendingChunk, None

    def clearChunk(self):
        self.readyChunk,self.pendingChunk = None, None

def makeBatch(args, dataloader, PH, idx=None, is_rand=False):
    if idx is None:
        raise('Error!')
    modelIdx = range(idx[0], idx[1]) 

    data = dataloader.readyChunk
    if is_rand:
        image_in_, pcgt_, depth_, mask_, K_, extrinsic_, r_ = PH
    else:
        image_in_, pcgt_, depth_, mask_, K_, extrinsic_ = PH

    image_in = data['image_in'][modelIdx]
    pcgt = data['pcgt'][modelIdx]
    depth = data['depth'][modelIdx]
    mask = data['mask'][modelIdx]
    extrinsic = data['extrinsic'][modelIdx]

    # ------ reshape input data ------
    image_in = np.reshape(image_in, [-1, args.in_H, args.in_W, 3])
    mask = np.ndarray.astype(mask, np.float32)
    mask = np.reshape(mask, [-1, args.depth_H, args.depth_W, 1])
    depth = np.reshape(depth, [-1, args.depth_H, args.depth_W, 1])
    pcgt = np.tile(np.expand_dims(pcgt, axis=1), [1, dataloader.num_images, 1, 1])
    pcgt = np.reshape(pcgt, [-1, args.pcgtSize, 3])
    K = np.tile(np.expand_dims(args.K, axis=0), [np.shape(pcgt)[0], 1, 1])
    extrinsic = np.reshape(extrinsic, [-1, 4, 4])

    if is_rand:
        r = np.random.normal(0, 1, args.rand_num*args.rand_dim*np.shape(image_in)[0])
        r = np.reshape(r, [args.rand_num, np.shape(image_in)[0], args.rand_dim])
        batch = {
            image_in_: image_in,
            pcgt_: pcgt,
            depth_: depth,
            mask_: mask,
            K_: K,
            extrinsic_: extrinsic,
            r_: r
        }
    else:
        batch = {
            image_in_: image_in,
            pcgt_: pcgt,
            depth_: depth,
            mask_: mask,
            K_: K,
            extrinsic_: extrinsic
        }
    return batch


def perform_test(args, sess, runList, PH, test_loader, stage=1, appendix=None):
    if args.cd_per_class == True:
        result_dic = {}
        for i in range(len(args.cat_list)):
            result_dic[args.cat_list[i]] = []
        args.batchSize_test = 1 # for cat13, we set batch size to 1 
 
    if stage == 2:
        r_fgsm, loss_fgsm, consis_loss_train_all = appendix 
    
    image_in, pcgt, depth, mask, K, extrinsic, r = PH

    NUM_ALL = test_loader.length()
    chunkN = int(np.ceil(float(NUM_ALL)/args.chunkSize_test))
    NUM_FINAL = NUM_ALL - (chunkN - 1) * args.chunkSize_test
    itPerChunk = args.chunkSize_test / args.batchSize_test

    test_loader.loadChunk(args, loadRange=[0, args.chunkSize_test])
    loss_list = []
    for c in range(chunkN):
        test_loader.shipChunk()
        if c != chunkN - 1:
            test_loader.thread = threading.Thread(target=test_loader.loadChunk,
                                                 args=[args, [(c+1)*args.chunkSize_test, min((c+2)*args.chunkSize_test, NUM_ALL)]])
            test_loader.thread.start()
        dataChunk = test_loader.readyChunk
  
        break_flag = 0
        for i in range(itPerChunk):
            # ------ make test batch ------
            if c != chunkN - 1:
                batch = makeBatch(args, test_loader, PH, [i*args.batchSize_test, (i+1)*args.batchSize_test], is_rand=True)
                batchSize_test = args.batchSize_test
            else:
                batch = makeBatch(args, test_loader, PH, [i*args.batchSize_test, min((i+1)*args.batchSize_test, NUM_FINAL)], is_rand=True)
                batchSize_test = min((i+1)*args.batchSize_test, NUM_FINAL) - (i*args.batchSize_test)
                if (i+1)*args.batchSize_test >= NUM_FINAL:
                    break_flag = 1

            if stage == 1:
                batch[r] = np.reshape(batch[r], (-1,args.rand_dim))
                batch[image_in] = np.tile(batch[image_in], (args.rand_num, 1, 1, 1))
            else:
                ## ------ heuristic search ------
                batch[r] = np.reshape(batch[r], (-1,args.rand_dim))
                r_batch = batch[r]
                extrinsic_batch = batch[extrinsic]
                batch[extrinsic] = np.tile(extrinsic_batch, (args.rand_num, 1, 1)) #[R*B*V,4,4]
                image_batch = batch[image_in]
                batch[image_in] = np.tile(image_batch, (args.rand_num, 1, 1, 1)) #[R*B*V,H,W,3]
                consis_loss_train_all_ = sess.run(consis_loss_train_all, feed_dict=batch) #[R*B*V*(V-1)/2,]
                consis_temp = np.mean(np.reshape(consis_loss_train_all_, (batchSize_test*args.rand_num, -1)), axis=1) #[R*B,V*(V-1)/2]
                consis_temp = np.reshape(consis_temp, (args.rand_num, -1)) #[R,B]
                consis_index = np.argmin(consis_temp, axis=0)
                r_list = []
                for j in range(batchSize_test):
                    r_list.append(r_batch[consis_index[j]*batchSize_test*args.num_images+j*args.num_images:consis_index[j]*batchSize_test*args.num_images+(j+1)*args.num_images,:])
  
                ## ------ FGSM ------
                batch[r] = np.concatenate(r_list, axis=0)
                batch[image_in] = image_batch #[B*N,H,W,3]
                batch[extrinsic] = extrinsic_batch #[B*N,4,4]
                for j in range(args.fgsm_iter):
                    batch[r], loss_fgsm_ = sess.run([r_fgsm, loss_fgsm], feed_dict=batch)
                    batch[r] = np.squeeze(batch[r])
            loss_batch = sess.run(runList, feed_dict=batch)
            if args.cd_per_class == True:
                name_temp = test_loader.data_list[c*args.chunkSize_test+i]
                cat_name = name_temp.split('/')[0]
                result_dic[cat_name].append(loss_batch)

            loss_list.append(np.array(loss_batch)*batchSize_test)
            if break_flag:
                break

        if c != chunkN - 1: 
            test_loader.thread.join()
    if args.cd_per_class == True:
        print(toBlue("CD_PER_CLASS=======================================================")) 
        result_sum = 0
        for key in result_dic:
            result_temp = result_dic[key]
            print len(result_temp)
            result_sum += len(result_temp)
            result_temp = np.array(result_temp)
            result = np.mean(result_temp, axis=0)
            cd_ = result[-4]
            cd_fps_ = result[-3]
            print '{}:  cd:{:.3f}   cd_fps:{:.3f}'.format(key, cd_, cd_fps_)
        print 'num:',result_sum
    loss_list = np.array(loss_list)
    return np.sum(loss_list, axis=0) / NUM_ALL

