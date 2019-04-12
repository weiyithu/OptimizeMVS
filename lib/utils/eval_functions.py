
import os, sys
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_dir, '..', 'external_ops', 'evaluation_metric'))
sys.path.append(os.path.join(this_dir, '..', 'external_ops', 'farthest_sampling'))
from tf_approxmatch import *
from tf_nndistance import *
from tf_sampling import *

def farthest_point_sampling(n, input_pc):
   '''
   Using Farthest Point Sampling Algorithm.
   Give input point cloud and downsample to point cloud which has n points.
   Params:
   -- n         : Expected output cloud's point number, eg: 1024
   -- input_pc  : Input point cloud. Tensor [batch_size, point_number, 3]
   
   Returns:
   -- output_pc : Output cloud which contains n points Tensor [batch_size, n, 3]
   '''  
   output_idx = farthest_point_sample(n, input_pc)
   output_pc = gather_point(input_pc, output_idx)
   return output_pc

def evaluation(gt, pred, is_emd=True):
    """
    Use EMD or CD to evaluate the difference between two 3D clouds
    Params:
    -- gt       : Groundtruth cloud. Tensor [batch_size, point_number1, 3]
    -- pred     : Prediction Cloud. Tensor [batch_size, point_number2, 3]
    -- is_emd   : Whether to use EMD.

    Returns:
    -- dist     : results. Tensor [batch_size, ]
    """

    if is_emd:
        dist = approx_match(gt, pred)
        return tf.reduce_mean(dist)
    else:
        dist1, dist2 = nn_distance(gt, pred) #dist1: gt->pred, dist2: pred->gt
        dist = dist1 + dist2
        return tf.reduce_mean(100*dist), tf.reduce_mean(100*dist1), tf.reduce_mean(100*dist2)


