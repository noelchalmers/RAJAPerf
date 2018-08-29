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

#include "POLYBENCH_2MM.hpp"

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

#define POLYBENCH_2MM_DATA_SETUP_HIP \
  Real_ptr tmp = m_tmp; \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type)); \
  allocAndInitHipDeviceData(tmp, m_tmp, m_ni * m_nj); \
  allocAndInitHipDeviceData(A, m_A, m_ni * m_nk); \
  allocAndInitHipDeviceData(B, m_B, m_nk * m_nj); \
  allocAndInitHipDeviceData(C, m_C, m_nj * m_nl); \
  allocAndInitHipDeviceData(D, m_D, m_ni * m_nl);


#define POLYBENCH_2MM_TEARDOWN_HIP \
  getHipDeviceData(m_D, D, m_ni * m_nl); \
  deallocHipDeviceData(tmp); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B); \
  deallocHipDeviceData(C); \
  deallocHipDeviceData(D);

__global__ void polybench_2mm_hip_1(Real_ptr tmp, Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_type alpha, Real_type beta, Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,j,k;
   if (ii < ni * nj) {
     *(tmp + ii) = 0.0;
     i = ii/nj; j = ii % nj;
     for (k=0; k < nk; k++) {
       POLYBENCH_2MM_BODY2;
     }
   }


}

__global__ void polybench_2mm_hip_2(Real_ptr tmp, Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_type alpha, Real_type beta, Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,l,j;
   if (ii < ni * nl) {
     *(D + ii) *= beta;
     i = ii/nl; l = ii % nl;
     for (j=0; j < nj; j++) {
       POLYBENCH_2MM_BODY4;
     }
   }
}


void POLYBENCH_2MM::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;


  if ( vid == Base_HIP ) {

    POLYBENCH_2MM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
      hipLaunchKernelGGL((polybench_2mm_hip_1), dim3(grid_size), dim3(block_size), 0, 0, tmp,A,B,C,D,alpha,beta,
                                                     m_ni,m_nj,m_nk,m_nl);

      memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
      initHipDeviceData(D,m_D,m_ni * m_nl );

      grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
      hipLaunchKernelGGL((polybench_2mm_hip_2), dim3(grid_size), dim3(block_size), 0, 0, tmp,A,B,C,D,alpha,beta,
                                                     m_ni,m_nj,m_nk,m_nl);

    }
    stopTimer();

    POLYBENCH_2MM_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_2MM_DATA_SETUP_HIP;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernel<
          RAJA::statement::For<0, RAJA::hip_block_exec,
            RAJA::statement::For<1, RAJA::hip_thread_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nj},
                                               RAJA::RangeSegment{0, nk}),
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_2MM_BODY1;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_2MM_BODY2;
        }
      );

      memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
      initHipDeviceData(D,m_D,m_ni * m_nl );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nj}),
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_2MM_BODY3;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_2MM_BODY4;
        }
      );

    }
    stopTimer();

    POLYBENCH_2MM_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_2MM : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

