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

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define DEL_DOT_VEC_2D_DATA_SETUP_HIP \
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr xdot; \
  Real_ptr ydot; \
  Real_ptr div; \
  Index_ptr real_zones; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  Real_ptr x1,x2,x3,x4 ; \
  Real_ptr y1,y2,y3,y4 ; \
  Real_ptr fx1,fx2,fx3,fx4 ; \
  Real_ptr fy1,fy2,fy3,fy4 ; \
\
  allocAndInitHipDeviceData(x, m_x, m_array_length); \
  allocAndInitHipDeviceData(y, m_y, m_array_length); \
  allocAndInitHipDeviceData(xdot, m_xdot, m_array_length); \
  allocAndInitHipDeviceData(ydot, m_ydot, m_array_length); \
  allocAndInitHipDeviceData(div, m_div, m_array_length); \
  allocAndInitHipDeviceData(real_zones, m_domain->real_zones, iend);

#define DEL_DOT_VEC_2D_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_div, div, m_array_length); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(xdot); \
  deallocHipDeviceData(ydot); \
  deallocHipDeviceData(div); \
  deallocHipDeviceData(real_zones);

__global__ void deldotvec2d(Real_ptr div,
                            const Real_ptr x1, const Real_ptr x2,
                            const Real_ptr x3, const Real_ptr x4,
                            const Real_ptr y1, const Real_ptr y2,
                            const Real_ptr y3, const Real_ptr y4,
                            const Real_ptr fx1, const Real_ptr fx2,
                            const Real_ptr fx3, const Real_ptr fx4,
                            const Real_ptr fy1, const Real_ptr fy2,
                            const Real_ptr fy3, const Real_ptr fy4,
                            const Index_ptr real_zones,
                            const Real_type half, const Real_type ptiny,
                            Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   if (ii < iend) {
     DEL_DOT_VEC_2D_BODY_INDEX;
     DEL_DOT_VEC_2D_BODY;
   }
}


void DEL_DOT_VEC_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = m_domain->n_real_zones;

  if ( vid == Base_HIP ) {

    DEL_DOT_VEC_2D_DATA_SETUP_HIP;

    NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
    NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
    NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
    NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((deldotvec2d), dim3(grid_size), dim3(block_size), 0, 0, div,
                                             x1, x2, x3, x4,
                                             y1, y2, y3, y4,
                                             fx1, fx2, fx3, fx4,
                                             fy1, fy2, fy3, fy4,
                                             real_zones,
                                             half, ptiny,
                                             iend);

    }
    stopTimer();

    DEL_DOT_VEC_2D_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    DEL_DOT_VEC_2D_DATA_SETUP_HIP;

    NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
    NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
    NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
    NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

    RAJA::ListSegment zones(m_domain->real_zones, m_domain->n_real_zones);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         zones, [=] __device__ (Index_type i) {
         DEL_DOT_VEC_2D_BODY;
       });

    }
    stopTimer();

    DEL_DOT_VEC_2D_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  DEL_DOT_VEC_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
