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

#include "POLYBENCH_3MM.hpp"

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

#define POLYBENCH_3MM_DATA_SETUP_HIP \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_ptr E = m_E; \
  Real_ptr F = m_F; \
  Real_ptr G = m_G; \
\
  allocAndInitHipDeviceData(A, m_A, m_ni * m_nk); \
  allocAndInitHipDeviceData(B, m_B, m_nk * m_nj); \
  allocAndInitHipDeviceData(C, m_C, m_nj * m_nm); \
  allocAndInitHipDeviceData(D, m_D, m_nm * m_nl); \
  allocAndInitHipDeviceData(E, m_E, m_ni * m_nj); \
  allocAndInitHipDeviceData(F, m_F, m_nj * m_nl); \
  allocAndInitHipDeviceData(G, m_G, m_ni * m_nl);


#define POLYBENCH_3MM_TEARDOWN_HIP \
  getHipDeviceData(m_G, G, m_ni * m_nl); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B); \
  deallocHipDeviceData(C); \
  deallocHipDeviceData(D); \
  deallocHipDeviceData(E); \
  deallocHipDeviceData(F); \
  deallocHipDeviceData(G);

__global__ void polybench_3mm_hip_1(Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_ptr E, Real_ptr F, Real_ptr G,
                       Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl, Index_type nm)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,j,k;
   if (ii < ni * nj) {
     *(E + ii) = 0.0;
     i = ii/nj; j = ii % nj;
     for (k=0; k < nk; k++) {
       POLYBENCH_3MM_BODY2;
     }
   }
}

__global__ void polybench_3mm_hip_2(Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_ptr E, Real_ptr F, Real_ptr G,
                       Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl, Index_type nm)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type j,l,m;
   if (ii < nj * nl) {
     *(F + ii) = 0.0;
     j = ii/nl; l = ii % nl;
     for (m=0; m < nm; m++) {
       POLYBENCH_3MM_BODY4;
     }
   }
}


__global__ void polybench_3mm_hip_3(Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_ptr E, Real_ptr F, Real_ptr G,
                       Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl, Index_type nm)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,l,j;
   if (ii < ni * nl) {
     *(G + ii) = 0.0;
     i = ii/nl; l = ii % nl;
     for (j=0; j < nj; j++) {
       POLYBENCH_3MM_BODY6;
     }
   }
}

void POLYBENCH_3MM::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;
  const Index_type nm = m_nm;


  if ( vid == Base_HIP ) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
      hipLaunchKernelGGL((polybench_3mm_hip_1), dim3(grid_size), dim3(block_size), 0, 0, A,B,C,D,E,F,G,
                                                     m_ni,m_nj,m_nk,m_nl,m_nm);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_nj * m_nl, block_size);
      hipLaunchKernelGGL((polybench_3mm_hip_2), dim3(grid_size), dim3(block_size), 0, 0, A,B,C,D,E,F,G,
                                                     m_ni,m_nj,m_nk,m_nl,m_nm);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
      hipLaunchKernelGGL((polybench_3mm_hip_3), dim3(grid_size), dim3(block_size), 0, 0, A,B,C,D,E,F,G,
                                                     m_ni,m_nj,m_nk,m_nl,m_nm);
    }
    stopTimer();


    POLYBENCH_3MM_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

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
          POLYBENCH_3MM_BODY1;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_3MM_BODY2;
        }

      );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nm}),
        [=] __device__ (Index_type j, Index_type l, Index_type m) {
          POLYBENCH_3MM_BODY3;
        },
        [=] __device__ (Index_type j, Index_type l, Index_type m) {
          POLYBENCH_3MM_BODY4;
        }

      );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nj}),
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_3MM_BODY5;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_3MM_BODY6;
        }

      );

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_3MM : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

