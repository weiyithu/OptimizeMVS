
import os, sys
import tensorflow as tf
import tflearn
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_dir, '..'))
import lib.backbone.single_img_encoder as graph
from lib.backbone.autoencoder import encoder, decoder, discriminator
from lib.utils.color_utils import *
from lib.utils.eval_functions import evaluation, farthest_point_sampling
from lib.utils.loss_functions import get_loss_diversity, front_loss, consis_loss, FGSM

def build_train_graph(args):
    print(toMagenta("building graph..."))
    tf.reset_default_graph()
    
    # ------ define graph ------
    with tf.device("/gpu:0"):
        ## ------ define input data ------
        r = tf.placeholder(tf.float32, shape=[None, args.rand_dim])
        image_in = tf.placeholder(tf.float32, shape=[None, args.in_H, args.in_W, 3])
        pcgt = tf.placeholder(tf.float32, shape=[None, args.pcgtSize, 3])
        depth = tf.placeholder(tf.float32, shape=[None, args.depth_H, args.depth_W, 1])
        mask = tf.placeholder(tf.float32, shape=[None, args.depth_H, args.depth_W, 1])
        K = tf.placeholder(tf.float32, shape=[None, 3, 3])
        extrinsic = tf.placeholder(tf.float32, shape=[None, 4, 4])
        PH = [image_in, pcgt, depth, mask, K, extrinsic, r]
    
        ## ------ build single-image encoder model ------
        tflearn.init_graph(seed=1029, num_cores=2, gpu_memory_fraction=0.9, soft_placement=True)
        pc  = graph.encoder(args, image_in, r)
        z_syn = encoder(pc)
        z_real = encoder(tf.tile(pcgt,[args.rand_num,1,1]),reuse=True)
        prob_syn, logit_syn = discriminator(z_syn)
        prob_real, logit_real = discriminator(z_real, reuse=True)
        train_vars = tf.trainable_variables()
        d_vars = [v for v in train_vars if 'discriminator' in v.name]
        g_vars = [v for v in train_vars if ('discriminator'  not in v.name and 'encoder' not in v.name)]
        e_vars = [v for v in train_vars if 'encoder' in v.name]
    
        ## ------ evaluation metric -----
        pcgt_s1 = tf.tile(pcgt,[args.rand_num,1,1]) #[R*B*V,N,3]
        pcgt_s2_train = tf.reshape(tf.tile(pcgt,[args.rand_num,1,1]),[-1, args.num_images, args.pcgtSize, 3])[:,0,:,:] #[R*B,N,3]
        pcgt_s2_test = tf.reshape(pcgt,[-1, args.num_images, args.pcgtSize, 3])[:,0,:,:] #[B,N,3]
        pc_s1 = pc #[R*B*V,N,3]
        pc_s2 = tf.reshape(pc, [-1, args.num_images*args.pcgtSize, 3]) #train: [R*B,N*V,3], test: [B,N*V,3]
    
        emd_s1 = evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=True)
        cd_s1, d1_s1, d2_s1 = evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=False)
        cd_fps_s2, d1_fps_s2, d2_fps_s2 = evaluation(gt=pcgt_s2_train, pred=farthest_point_sampling(args.pcgtSize, pc_s2), is_emd=False)
        cd_s2, d1_s2, d2_s2 = evaluation(gt=pcgt_s2_train, pred=pc_s2, is_emd=False)
        cd_fps_s2_evaluate, d1_fps_s2_evaluate, d2_fps_s2_evaluate = evaluation(gt=pcgt_s2_test, pred=farthest_point_sampling(args.pcgtSize, pc_s2), is_emd=False)
        cd_s2_evaluate, d1_s2_evaluate, d2_s2_evaluate = evaluation(gt=pcgt_s2_test, pred=pc_s2, is_emd=False)
    
        ## ------ train loss ------
        ### ------ diversity constraint ------
        r_reshape = tf.reshape(r, shape=[args.rand_num, -1, args.rand_dim]) #[R,B*V,D]
        pc_reshape = tf.reshape(pc,shape=[args.rand_num, -1, args.pcgtSize, 3]) #[R,B*V,N,3]
        loss_div_s1 = get_loss_diversity(r_reshape, pc_reshape, args.rand_num, alpha=args.alpha_s1, is_emd=True)
        loss_div_s2 = get_loss_diversity(r_reshape, pc_reshape, args.rand_num, alpha=args.alpha_s2, is_emd=True)
    
        ### ------ front constraint ------
        K_tile = tf.tile(K, [args.rand_num, 1, 1]) #[R*B*V,3,3]
        extrinsic_tile = tf.tile(extrinsic, [args.rand_num, 1, 1]) #[R*B*V,4,4]
        depth_tile = tf.tile(tf.squeeze(depth, [3]), [args.rand_num, 1, 1]) #[R*B*V,H,W]
        loss_front_s1 = front_loss(pc, K_tile, extrinsic_tile, depth_tile, H=args.depth_H/args.scale, W=args.depth_W/args.scale, reuse=True)
        pc_concat = tf.reshape(pc, [-1, args.num_images*args.pcgtSize,3])
        pc_concat = tf.reshape(tf.tile(tf.expand_dims(pc_concat,axis=1), [1,args.num_images, 1,1]),[-1, args.num_images*args.pcgtSize, 3])
        loss_front_s2 = front_loss(pc_concat, K_tile, extrinsic_tile, depth_tile,  H=args.depth_H/args.scale, W=args.depth_W/args.scale, reuse=True, is_emd=False)
    
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
    
    
        loss_s1 = loss_front_s1 + args.loss_weight_div_s1*loss_div_s1  + args.loss_weight_adv*g_loss
        loss_s2 = loss_front_s2 + args.loss_weight_div_s2*loss_div_s2  + args.loss_weight_adv*g_loss
        loss_d = args.loss_weight_adv*d_loss
    
        ## ------ evaluation loss (only for stage2) ------
        loss_front_s2_evaluate = front_loss(pc_concat, K, extrinsic, tf.squeeze(depth, [3]), H=args.depth_H/args.scale, W=args.depth_W/args.scale, reuse=True, is_emd=False)
    
        ## ------ consistency loss (only for inference) ------
        extrinsic_consis = tf.tile(tf.expand_dims(tf.eye(4), axis=0), [tf.shape(pc)[0], 1, 1])
        consis_loss_all_cd = consis_loss(tf.reshape(pc, [-1, args.num_images, args.pcgtSize, 3]), tf.reshape(extrinsic_consis, [-1, args.num_images, 4, 4]), args.num_images, is_emd=False)
        consis_loss_cd = tf.reduce_mean(consis_loss_all_cd)
        r_fgsm, loss_fgsm = FGSM(r, consis_loss_cd, args.fgsm_lr) 
    
    
        ## ------ optimizer ------
        lr_PH = tf.placeholder(tf.float32, shape=[])
        optim_s1 = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss_s1, var_list=g_vars)
        optim_s2 = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss_s2, var_list=g_vars)
        optim_d = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss_d, var_list=d_vars)

        # train output
        ret = {}
        ret['PH'] = PH
        ret['e_vars'] = e_vars

        ret['optim_s1'] = optim_s1
        ret['loss_s1'] = loss_s1
        ret['cd_s1'] = cd_s1
        ret['emd_s1'] = emd_s1
        ret['loss_front_s1'] = loss_front_s1
        ret['loss_div_s1'] = loss_div_s1
        ret['g_loss'] = g_loss
        ret['d1_s1'] = d1_s1
        ret['d2_s1'] = d2_s1

        ret['optim_s2'] = optim_s2
        ret['loss_s2'] = loss_s2
        ret['cd_s2'] = cd_s2
        ret['cd_fps_s2'] = cd_fps_s2
        ret['loss_front_s2'] = loss_front_s2
        ret['loss_div_s2'] = loss_div_s2
        ret['loss_front_s2_evaluate'] = loss_front_s2_evaluate
        ret['cd_s2_evaluate'] = cd_s2_evaluate
        ret['cd_fps_s2_evaluate'] = cd_fps_s2_evaluate
        ret['d1_fps_s2_evaluate'] = d1_fps_s2_evaluate
        ret['d2_fps_s2_evaluate'] = d2_fps_s2_evaluate

        ret['optim_d'] = optim_d
        ret['d_loss'] = d_loss
        ret['r_fgsm'] = r_fgsm
        ret['loss_fgsm'] = loss_fgsm
        ret['consis_loss_all_cd'] = consis_loss_all_cd

        ret['lr_PH'] = lr_PH
        ret['r'] = r
        ret['image_in'] = image_in
        return ret


def build_test_graph(args):
    print(toMagenta("building graph..."))
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
    
        emd_s1 = evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=True)
        cd_s1, d1_s1, d2_s1 = evaluation(gt=pcgt_s1, pred=pc_s1, is_emd=False)
        cd_fps_s2_evaluate, d1_fps_s2_evaluate, d2_fps_s2_evaluate = evaluation(gt=pcgt_s2_test, pred=farthest_point_sampling(opt.pcgtSize, pc_s2), is_emd=False)
        cd_s2_evaluate, d1_s2_evaluate, d2_s2_evaluate = evaluation(gt=pcgt_s2_test, pred=pc_s2, is_emd=False)
    
        ## ------ evaluation loss ------
        K_tile = tf.tile(K, [opt.rand_num, 1, 1]) #[R*B*V,3,3]
        extrinsic_tile = tf.tile(extrinsic, [opt.rand_num, 1, 1]) #[R*B*V,4,4]
        depth_tile = tf.tile(tf.squeeze(depth, [3]), [opt.rand_num, 1, 1]) #[R*B*V,H,W]
        loss_front_s1 = front_loss(pc, K_tile, extrinsic_tile, depth_tile, H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True)
        pc_concat = tf.reshape(pc, [-1, opt.num_images*opt.pcgtSize,3])
        pc_concat = tf.reshape(tf.tile(tf.expand_dims(pc_concat,axis=1), [1,opt.num_images, 1,1]),[-1, opt.num_images*opt.pcgtSize, 3])
        loss_front_s2_evaluate = front_loss(pc_concat, K, extrinsic, tf.squeeze(depth, [3]), H=opt.depth_H/opt.scale, W=opt.depth_W/opt.scale, reuse=True, is_emd=False)
    
        ## ------ inference loss ------
        extrinsic_consis = tf.tile(tf.expand_dims(tf.eye(4), axis=0), [tf.shape(pc)[0], 1, 1])
        
        consis_loss_all_cd = consis_loss(tf.reshape(pc, [-1, opt.num_images, opt.pcgtSize, 3]), tf.reshape(extrinsic_consis, [-1, opt.num_images, 4, 4]), opt.num_images, is_emd=False)
        consis_loss_cd = tf.reduce_mean(consis_loss_all_cd)
        r_fgsm, loss_fgsm = FGSM(r, consis_loss_cd, opt.fgsm_lr) 
    
    
        ## ------ optimizer ------
        lr_PH = tf.placeholder(tf.float32, shape=[])

        # test output
        ret = {}
        ret['PH'] = PH

        ret['loss_front_s1'] = loss_front_s1
        ret['cd_s1'] = cd_s1
        ret['emd_s1'] = emd_s1
        ret['g_loss'] = g_loss
        ret['d1_s1'] = d1_s1
        ret['d2_s1'] = d2_s1

        ret['loss_front_s2_evaluate'] = loss_front_s2_evaluate
        ret['cd_s2_evaluate'] = cd_s2_evaluate
        ret['cd_fps_s2_evaluate'] = cd_fps_s2_evaluate
        ret['d1_fps_s2_evaluate'] = d1_fps_s2_evaluate
        ret['d2_fps_s2_evaluate'] = d2_fps_s2_evaluate

        ret['r_fgsm'] = r_fgsm
        ret['loss_fgsm'] = loss_fgsm
        ret['consis_loss_all_cd'] = consis_loss_all_cd

        ret['lr_PH'] = lr_PH
        ret['r'] = r
        ret['image_in'] = image_in
        return ret


