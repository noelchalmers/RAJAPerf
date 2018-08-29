#include "hip/hip_runtime.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMMVER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

//
// Define thread block size for HIP execution
//
const size_t block_size = 256;

#define POLYBENCH_GEMMVER_DATA_SETUP_HIP \
  Index_type n = m_n; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  Real_ptr A = m_A; \
  Real_ptr u1 = m_u1; \
  Real_ptr v1 = m_v1; \
  Real_ptr u2 = m_u2; \
  Real_ptr v2 = m_v2; \
  Real_ptr w = m_w; \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
\
  allocAndInitHipDeviceData(A, m_A, m_n * m_n); \
  allocAndInitHipDeviceData(u1, m_u1, m_n); \
  allocAndInitHipDeviceData(v1, m_v1, m_n); \
  allocAndInitHipDeviceData(u2, m_u2, m_n); \
  allocAndInitHipDeviceData(v2, m_v2, m_n); \
  allocAndInitHipDeviceData(w, m_w, m_n); \
  allocAndInitHipDeviceData(x, m_x, m_n); \
  allocAndInitHipDeviceData(y, m_y, m_n); \
  allocAndInitHipDeviceData(z, m_z, m_n);


#define POLYBENCH_GEMMVER_TEARDOWN_HIP \
  getHipDeviceData(m_w, w, m_n); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(u1); \
  deallocHipDeviceData(v1); \
  deallocHipDeviceData(u2); \
  deallocHipDeviceData(v2); \
  deallocHipDeviceData(w); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(z);

__global__ void polybench_gemmver_hip_1(Real_ptr A,
                       Real_ptr u1, Real_ptr v1, Real_ptr u2,
                       Real_ptr v2, Index_type n)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,j;
   if (ii < n * n) {
     i = ii/n; j = ii % n;
     POLYBENCH_GEMMVER_BODY1;
   }
}

__global__ void polybench_gemmver_hip_2(Real_type beta,
                       Real_ptr A, Real_ptr x, Real_ptr y,
                       Index_type n)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,jj;
   if (ii < n * n) {
     i = ii/n; jj = ii % n;
     if (jj == 0) {
       for(Index_type j=0; j < n; ++j) {
         POLYBENCH_GEMMVER_BODY2;
       }
     }

   }
}


__global__ void polybench_gemmver_hip_3(Real_ptr x,
                       Real_ptr z, Real_ptr v1, Real_ptr u2,
                       Real_ptr v2, Index_type n)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     POLYBENCH_GEMMVER_BODY3;
   }
}

__global__ void polybench_gemmver_hip_4(Real_type alpha,
                       Real_ptr A, Real_ptr x, Real_ptr w,
                       Index_type n)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,jj;
   if (ii < n * n) {
     i = ii/n; jj = ii % n;
     if (jj == 0) {
       for(Index_type j=0; j < n; ++j) {
         POLYBENCH_GEMMVER_BODY4;
       }
     }
   }
}



void POLYBENCH_GEMMVER::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_HIP ) {

    POLYBENCH_GEMMVER_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
      hipLaunchKernelGGL((polybench_gemmver_hip_1), dim3(grid_size), dim3(block_size), 0, 0, A,u1,v1,u2,v2,n);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
      hipLaunchKernelGGL((polybench_gemmver_hip_2), dim3(grid_size), dim3(block_size), 0, 0, beta,A,x,y,n);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_n , block_size);
      hipLaunchKernelGGL((polybench_gemmver_hip_3), dim3(grid_size), dim3(block_size), 0, 0, x,z,v1,u2,v2,n);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
      hipLaunchKernelGGL((polybench_gemmver_hip_4), dim3(grid_size), dim3(block_size), 0, 0, alpha,A,x,w,n);

    }
    stopTimer();

    POLYBENCH_GEMMVER_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GEMMVER_DATA_SETUP_HIP;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::hip_exec<block_size, async>> (
        RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
        Index_type i,j;
        i = ii/n; j = ii % n;
        POLYBENCH_GEMMVER_BODY1;
      });

      RAJA::forall<RAJA::hip_exec<block_size, async>> (
        RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,jj;
          i = ii/n; jj = ii % n;
          if (jj == 0) {
            for(Index_type j=0; j < n; ++j) {
              POLYBENCH_GEMMVER_BODY2;
            }
          }
      });

      RAJA::forall<RAJA::hip_exec<block_size, async>> (
        RAJA::RangeSegment{0, n}, [=] __device__ (int i) {
        POLYBENCH_GEMMVER_BODY3;
      });

      RAJA::forall<RAJA::hip_exec<block_size, async>> (
        RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,jj;
          i = ii/n; jj = ii % n;
          if (jj == 0) {
            for(Index_type j=0; j < n; ++j) {
              POLYBENCH_GEMMVER_BODY4;
            }
          }
      });

    }
    stopTimer();

    POLYBENCH_GEMMVER_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_GEMMVER : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

