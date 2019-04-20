# build evaluation metric
cd lib/external_ops/evaluation_metric
sh tf_approxmatch_compile.sh
sh tf_nndistance_compile.sh

# build farthest sampling
cd ../farthest_sampling
sh tf_sampling_compile.sh
cd ../../..

