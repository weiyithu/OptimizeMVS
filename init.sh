# set datadir to the shapenet directory.
# datadir=/home/data/shapenet
# ln -s $datadir data

## download data
mkdir -p data && cd data
# download point clouds
wget https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=0
mv shape_net_core_uniform_samples_2048.zip\?dl\=0 shape_net_core_uniform_samples_2048.zip
unzip shape_net_core_uniform_samples_2048.zip
mv shape_net_core_uniform_samples_2048 pcdata_2048
rm shape_net_core_uniform_samples_2048.zip
# download rendered multi-view images
# TODO

## build evaluation metric
cd ../lib/external_ops/evaluation_metric
sh tf_approxmatch_compile.sh
sh tf_nndistance_compile.sh

## build farthest sampling
cd ../farthest_sampling
sh tf_sampling_compile.sh
cd ../../..

## download autoencoder model
wget ftp://93.179.103.61/OptimizeMVS/pretrain.tar.gz
tar zxf pretrain.tar.gz

