
import os, sys
import numpy as np
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_dir)
from parse_args import parse_args

CAT_LIST_1 = ['03001627']
CAT_LIST_13 = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
CAMERA_INTRINSIC = np.array([[420., 0., 112.],
                            [0., 420., 112.],
                            [0., 0., 1.]])

def init_config():
    args = parse_args()
    # below automatically set
    args.in_H, args.in_W = [int(x) for x in args.inSize.split("x")]
    args.depth_H, args.depth_W = [int(x) for x in args.depthSize.split("x")]

    args.datalist_path = os.path.join(args.dataset_path, args.datalist_path)
    args.pcgt_path = os.path.join(args.dataset_path, args.pcgt_path)
    args.rendering_path = os.path.join(args.dataset_path, args.rendering_path)
    
    if args.cat == 1:
        args.cat_list = CAT_LIST_1
        args.test_iter = 2000
    elif args.cat == 13:
        args.cat_list = CAT_LIST_13
        args.test_iter = 5000
    else:
        raise NotImplementedError 

    args.gpu_id = str(args.gpu)

    # below constant
    args.K = CAMERA_INTRINSIC
    return args

