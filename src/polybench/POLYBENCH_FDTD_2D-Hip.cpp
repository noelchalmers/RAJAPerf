  
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

#include "POLYBENCH_FDTD_2D.hpp"

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

#define POLYBENCH_FDTD_2D_DATA_SETUP_HIP \
  const Index_type nx = m_nx; \
  const Index_type ny = m_ny; \
  const Index_type tsteps = m_tsteps; \
\
  Real_ptr fict; \
  Real_ptr ex; \
  Real_ptr ey; \
  Real_ptr hz; \
\
  allocAndInitHipDeviceData(hz, m_hz, m_nx * m_ny); \
  allocAndInitHipDeviceData(ex, m_ex, m_nx * m_ny); \
  allocAndInitHipDeviceData(ey, m_ey, m_nx * m_ny); \
  allocAndInitHipDeviceData(fict, m_fict, m_tsteps);


#define POLYBENCH_FDTD_2D_TEARDOWN_HIP \
  getHipDeviceData(m_hz, hz, m_nx * m_ny); \
  deallocHipDeviceData(ex); \
  deallocHipDeviceData(ey); \
  deallocHipDeviceData(fict);


__global__ void poly_fdtd2d_1(Real_ptr ey, Real_ptr fict,
                              Index_type ny, Index_type t)
{
   Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

   if (j < ny) {
     POLYBENCH_FDTD_2D_BODY1;
   }
}

__global__ void poly_fdtd2d_2(Real_ptr ey, Real_ptr hz, Index_type ny)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   if (i > 0) {
     POLYBENCH_FDTD_2D_BODY2;
   }
}

__global__ void poly_fdtd2d_3(Real_ptr ex, Real_ptr hz, Index_type ny)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   if (j > 0) {
     POLYBENCH_FDTD_2D_BODY3;
   }
}

__global__ void poly_fdtd2d_4(Real_ptr hz, Real_ptr ex, Real_ptr ey,
                              Index_type nx, Index_type ny)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   if (i < nx-1 && j < ny-1) {
     POLYBENCH_FDTD_2D_BODY4;
   }
}



void POLYBENCH_FDTD_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_HIP ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);
        hipLaunchKernelGGL((poly_fdtd2d_1), dim3(grid_size1), dim3(block_size), 0, 0, ey, fict, ny, t);

        dim3 nblocks234(nx, 1, 1);
        dim3 nthreads_per_block234(1, ny, 1);
        hipLaunchKernelGGL((poly_fdtd2d_2), dim3(nblocks234), dim3(nthreads_per_block234), 
                                    0, 0, ey, hz, ny);

        hipLaunchKernelGGL((poly_fdtd2d_3), dim3(nblocks234), dim3(nthreads_per_block234), 
                                    0, 0, ex, hz, ny);

        hipLaunchKernelGGL((poly_fdtd2d_4), dim3(nblocks234), dim3(nthreads_per_block234), 
                                    0, 0, hz, ex, ey, nx, ny);

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::hip_exec<block_size, true /*async*/>;

    using EXEC_POL234 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_exec,
            RAJA::statement::For<1, RAJA::hip_thread_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny),
         [=] __device__ (Index_type j) {
           POLYBENCH_FDTD_2D_BODY1_RAJA;
        });

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                           RAJA::RangeSegment{0, ny}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                           RAJA::RangeSegment{1, ny}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                           RAJA::RangeSegment{0, ny-1}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4_RAJA;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_FDTD_2D : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
  
