#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

rm -rf build_ubuntu_nvcc_gcc-coral-2018.04.31 >/dev/null
mkdir build_ubuntu_nvcc_gcc-coral-2018.04.31 && cd build_ubuntu_nvcc_gcc-coral-2018.04.31

#module load cmake/3.7.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/ubuntu/nvcc_gcc_coral_2018_08_31.cmake \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_ubuntu_nvcc_gcc-coral-2018.04.31 \
  "$@" \
  ..
