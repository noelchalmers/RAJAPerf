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

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_OFFSET_DATA_SETUP_HIP \
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitHipDeviceData(a, m_a, iend);

#define INIT_VIEW1D_OFFSET_DATA_RAJA_SETUP_HIP \
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitHipDeviceData(a, m_a, iend); \
\
  using ViewType = RAJA::View<Real_type, RAJA::OffsetLayout<1> >; \
  ViewType view(a, RAJA::make_offset_layout<1>({{1}}, {{iend+1}}));

#define INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_a, a, iend); \
  deallocHipDeviceData(a);

__global__ void initview1d_offset(Real_ptr a,
                                  Real_type v,
                                  const Index_type ibegin,
                                  const Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INIT_VIEW1D_OFFSET_BODY;
   }
}


void INIT_VIEW1D_OFFSET::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getRunSize()+1;

  if ( vid == Base_HIP ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((initview1d_offset), dim3(grid_size), dim3(block_size), 0, 0,  a, v,
                                                    ibegin,
                                                    iend );

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    INIT_VIEW1D_OFFSET_DATA_RAJA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY_RAJA;
      });

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
