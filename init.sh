# TODO: set datadir to the shapenet directory.
# datadir=/home/data/shapenet
# ln -s $datadir data

# download data


# build evaluation metric
cd lib/external_ops/evaluation_metric
sh tf_approxmatch_compile.sh
sh tf_nndistance_compile.sh

# build farthest sampling
cd ../farthest_sampling
sh tf_sampling_compile.sh
cd ../../..

# download autoencoder model
wget ftp://93.179.103.61/OptimizeMVS/pretrain.tar.gz
tar zxf pretrain.tar.gz

