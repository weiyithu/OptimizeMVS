TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
${1}"/bin/nvcc" -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I ${2}"/site-packages/tensorflow/include" -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2   && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I ${2}"/site-packages/tensorflow/include" -L ${1}"/lib64" -O2 
