  
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
// For details about use and distribution, please read RAJAPerf/LICENSE.
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

__global__ void poly_3mm_1(Real_ptr E, Real_ptr A, Real_ptr B,
                           Index_type nj, Index_type nk)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   POLYBENCH_3MM_BODY1;

   for (Index_type k=0; k < nk; ++k) {
     POLYBENCH_3MM_BODY2;
   }
}

__global__ void poly_3mm_2(Real_ptr F, Real_ptr C, Real_ptr D,
                           Index_type nl, Index_type nm)
{
   Index_type j = blockIdx.x;
   Index_type l = threadIdx.y;

   POLYBENCH_3MM_BODY3;

   for (Index_type m=0; m < nm; ++m) {
     POLYBENCH_3MM_BODY4;
   }
}

__global__ void poly_3mm_3(Real_ptr G, Real_ptr E, Real_ptr F,
                           Index_type nl, Index_type nj)
{
   Index_type i = blockIdx.x;
   Index_type l = threadIdx.y;

   POLYBENCH_3MM_BODY5;

   for (Index_type j=0; j < nj; ++j) {
     POLYBENCH_3MM_BODY6;
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

      dim3 nblocks1(ni, 1, 1);
      dim3 nthreads_per_block1(1, nj, 1);
      hipLaunchKernelGGL((poly_3mm_1), dim3(nblocks1) , dim3(nthreads_per_block1), 0, 0, 
                                                    E, A, B,
                                                    nj, nk);

      dim3 nblocks2(nj, 1, 1);
      dim3 nthreads_per_block2(1, nl, 1);
      hipLaunchKernelGGL((poly_3mm_2), dim3(nblocks2), dim3(nthreads_per_block2), 0, 0,
                                                    F, C, D,
                                                    nl, nm);

      dim3 nblocks3(ni, 1, 1);
      dim3 nthreads_per_block3(1, nl, 1);
      hipLaunchKernelGGL((poly_3mm_3), dim3(nblocks3), dim3(nthreads_per_block3), 0, 0, 
                                                    G, E, F,
                                                    nl, nj);

    }
    stopTimer();
    
    POLYBENCH_3MM_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    POLYBENCH_3MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
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
        [=] __device__ (Index_type i, Index_type j, Index_type /* k */) {
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_3MM_BODY2_RAJA;
        }

      );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nm}),
        [=] __device__ (Index_type j, Index_type l, Index_type /* m */) {
          POLYBENCH_3MM_BODY3_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l, Index_type m) {
          POLYBENCH_3MM_BODY4_RAJA;
        }

      );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nj}),
        [=] __device__ (Index_type i, Index_type l, Index_type /* j */) {
          POLYBENCH_3MM_BODY5_RAJA;
        }, 
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_3MM_BODY6_RAJA;
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
  
