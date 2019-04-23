#cuda and python path
cuda_path=/usr/local/cuda-8.0
py_path=~/.local/lib/python2.7

# build evaluation metric
cd lib/external_ops/evaluation_metric
sh tf_approxmatch_compile.sh $cuda_path $py_path
sh tf_nndistance_compile.sh $cuda_path $py_path

# build farthest sampling
cd ../farthest_sampling
sh tf_sampling_compile.sh $cuda_path $py_path
cd ../../..


