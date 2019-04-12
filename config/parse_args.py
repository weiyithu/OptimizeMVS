
import argparse

DATASET_PATH = '/data2/weiy/data-3d/shapenet/'

def parse_args():
    parser = argparse.ArgumentParser()
   
    # ------ basic setting ------
    parser.add_argument("--gpu", type=int, required=True, help="gpu_id")
    parser.add_argument('--task', default="bingo", help='task name, to be appeared at the title of the diary.')
    parser.add_argument('--group', default="0", help='group name')
    parser.add_argument('--model', default="bingo", help='model name')
    parser.add_argument("--load", default=None,	help="load trained model to fine-tune/evaluate")
    parser.add_argument("--cat", type=int, default=1, help="number of categories (1 or 13)")
    parser.add_argument("--data_use_mask", type=bool, default=False, help="whether use mask for data processing")
    parser.add_argument("--inSize", default="224x224", help="resolution of encoder input")
    parser.add_argument("--depthSize", default="224x224", help="resolution of depth images")
    parser.add_argument("--pcgtSize", type=int, default=2048, help="size of point clouds (groundtruth)")
    parser.add_argument("--scale",   type=int, default=4,  help="the ratio size between RGB image(input) and depth image(supervised)")
    parser.add_argument("--num_images", type=int, default=8, help="number of input imagaes per object")
    parser.add_argument("--rand_num", type=int, default=5, help="number of random inputs per image")
    parser.add_argument("--rand_dim", type=int, default=128, help="the dimension of each random input")
    parser.add_argument("--stage", type=int, default=2, help="stage for evaluation")

    # ------ path ------
    parser.add_argument('--savedir', default="output", help='model save directory')
    parser.add_argument('--diary_path', default="diary", help='diary path')
    parser.add_argument('--dataset_path', default=DATASET_PATH, help='dataset path')
    parser.add_argument('--datalist_path', default='data_list', help='relative path of data list w.r.t dataset path')
    parser.add_argument('--pcgt_path', default='pcdata_2048', help='relative path of point cloud data w.r.t dataset path')
    parser.add_argument('--rendering_path', default='rendering_uniform', help='relative path of rendered data w.r.t dataset path')

    # ------ hyper-parameter (train) ------
    parser.add_argument("--alpha_s1", type=float, default=0.2,  help="diversity loss alpha for stage1")
    parser.add_argument("--alpha_s2", type=float, default=0.1,  help="diversity loss alpha for stage2")
    parser.add_argument("--batchSize", type=int, default=2,   help="batch size for training")
    parser.add_argument("--chunkSize", type=int, default=100,  help="data chunk size to load")
    parser.add_argument("--lr",   type=float, default=1e-4,  help="base learning rate ")
    parser.add_argument("--lrDecay", type=float, default=1.0,  help="learning rate decay multiplier")
    parser.add_argument("--lrStep",  type=int, default=20000,  help="learning rate decay step size")
    parser.add_argument("--fromIt_stage1",  type=int, default=0,   help="resume training from iteration number for stage1")
    parser.add_argument("--fromIt_stage2",  type=int, default=0,   help="resume training from iteration number for stage2 ")
    parser.add_argument("--toIt_stage1",  type=int, default=40000,  help="run training to iteration number for stage1")
    parser.add_argument("--toIt_stage2",  type=int, default=100000,  help="run training to iteration number for stage2")
    parser.add_argument("--loss_weight_div_s1",   type=float, default=10,  help="loss weight of diversity loss for stage1")
    parser.add_argument("--loss_weight_div_s2",   type=float, default=1.0,  help="loss weight of diversity loss for stage2")
    parser.add_argument("--loss_weight_adv",   type=float, default=0.1,  help="loss weight of adversial loss")

    # ------ hyper-parameter (test) ------
    parser.add_argument("--cd_per_class", type=bool, default=False, help="whether evaluate cd for every class (only for cat13)")
    parser.add_argument("--fgsm_iter", type=int, default=10, help="the iterations of fgsm")
    parser.add_argument("--fgsm_lr",   type=float, default=1,  help="base learning rate ")
    parser.add_argument("--batchSize_test", type=int, default=2,  help="batch size for evaluation")
    parser.add_argument("--chunkSize_test", type=int, default=100,  help="data chunk size to load for evaluation")

    args = parser.parse_args()
    return args
 
