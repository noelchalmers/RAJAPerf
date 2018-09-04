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

#include "INIT3.hpp"

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


#define INIT3_DATA_SETUP_HIP \
  Real_ptr out1; \
  Real_ptr out2; \
  Real_ptr out3; \
  Real_ptr in1; \
  Real_ptr in2; \
\
  allocAndInitHipDeviceData(out1, m_out1, iend); \
  allocAndInitHipDeviceData(out2, m_out2, iend); \
  allocAndInitHipDeviceData(out3, m_out3, iend); \
  allocAndInitHipDeviceData(in1, m_in1, iend); \
  allocAndInitHipDeviceData(in2, m_in2, iend);

#define INIT3_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_out1, out1, iend); \
  getHipDeviceData(m_out2, out2, iend); \
  getHipDeviceData(m_out3, out3, iend); \
  deallocHipDeviceData(out1); \
  deallocHipDeviceData(out2); \
  deallocHipDeviceData(out3); \
  deallocHipDeviceData(in1); \
  deallocHipDeviceData(in2);

__global__ void init3(Real_ptr out1, Real_ptr out2, Real_ptr out3,
                      Real_ptr in1, Real_ptr in2,
                      Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INIT3_BODY;
   }
}


void INIT3::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_HIP ) {

    INIT3_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((init3), dim3(grid_size), dim3(block_size), 0, 0,  out1, out2, out3, in1, in2,
                                        iend );

    }
    stopTimer();

    INIT3_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    INIT3_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT3_BODY;
      });

    }
    stopTimer();

    INIT3_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  INIT3 : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP