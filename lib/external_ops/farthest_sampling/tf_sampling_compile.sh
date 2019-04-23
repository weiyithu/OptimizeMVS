#/bin/bash
${1}"/bin/nvcc" tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${2}"/site-packages/tensorflow/include" -I ${1}"/include" -lcudart -L ${1}"/lib64/" -O2 -I $TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework

