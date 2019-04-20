# set datadir to the shapenet directory.
# datadir=/home/data/shapenet
# ln -s $datadir data

# download data
mkdir -p data && cd data
## download point clouds
wget https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=0
mv shape_net_core_uniform_samples_2048.zip\?dl\=0 shape_net_core_uniform_samples_2048.zip
unzip shape_net_core_uniform_samples_2048.zip
mv shape_net_core_uniform_samples_2048 pcdata_2048
rm shape_net_core_uniform_samples_2048.zip
## download rendered multi-view images
wget https://image.moeclub.org/GoogleDrive/1yWz1k0RG-KW8DhjezM1JSOgapLMf91A2
mv 1yWz1k0RG-KW8DhjezM1JSOgapLMf91A2 rendered-shapenet.tar.gz
tar zxf rendered-shapenet.tar.gz
rm rendered-shapenet.tar.gz 

## download data list
wget https://image.moeclub.org/GoogleDrive/1-gRyvYD2RjfdpONGUBYj20gCsTIUJ82r
mv 1-gRyvYD2RjfdpONGUBYj20gCsTIUJ82r data_list.tar.gz
tar zxf data_list.tar.gz
rm data_list.tar.gz

# download autoencoder model
cd ..
wget https://image.moeclub.org/GoogleDrive/1YvGhVYF-o9pA3u4fGAspF2UlpohV6zoy
mv 1YvGhVYF-o9pA3u4fGAspF2UlpohV6zoy pretrain.tar.gz
tar zxf pretrain.tar.gz
rm pretrain.tar.gz

