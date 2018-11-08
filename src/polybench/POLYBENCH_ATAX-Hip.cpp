  
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

#include "POLYBENCH_ATAX.hpp"

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

#define POLYBENCH_ATAX_DATA_SETUP_HIP \
  Real_ptr tmp; \
  Real_ptr y; \
  Real_ptr x; \
  Real_ptr A; \
\
  allocAndInitHipDeviceData(tmp, m_tmp, N); \
  allocAndInitHipDeviceData(y, m_y, N); \
  allocAndInitHipDeviceData(x, m_x, N); \
  allocAndInitHipDeviceData(A, m_A, N * N);


#define POLYBENCH_ATAX_TEARDOWN_HIP \
  getHipDeviceData(m_y, y, N); \
  deallocHipDeviceData(tmp); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(A);


__global__ void poly_atax_1(Real_ptr A, Real_ptr x, Real_ptr y, Real_ptr tmp,
                            Index_type N)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < N) { 
     POLYBENCH_ATAX_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_ATAX_BODY2;
     }
   }
}

__global__ void poly_atax_2(Real_ptr A, Real_ptr tmp, Real_ptr y,
                            Index_type N)
{
   Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

   if (j < N) { 
     for (Index_type i = 0; i < N; ++i ) {
       POLYBENCH_ATAX_BODY3;
     }
   }
}


void POLYBENCH_ATAX::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  if ( vid == Base_HIP ) {

    POLYBENCH_ATAX_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      hipLaunchKernelGGL((poly_atax_1), dim3(grid_size), dim3(block_size), 0, 0,
                                      A, x, y, tmp, N);

      hipLaunchKernelGGL((poly_atax_2), dim3(grid_size), dim3(block_size), 0, 0,
                                      A, tmp, y, N);

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_ATAX_DATA_SETUP_HIP;

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_threadblock_exec<block_size>,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >,
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<1, RAJA::hip_threadblock_exec<block_size>,
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<2>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N}),
        [=] __device__ (Index_type i, Index_type /* j */) {
          POLYBENCH_ATAX_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_ATAX_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_ATAX_BODY3_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_ATAX : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
  
