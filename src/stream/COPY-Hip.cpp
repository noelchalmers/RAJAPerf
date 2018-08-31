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

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define COPY_DATA_SETUP_HIP \
  Real_ptr a; \
  Real_ptr c; \
\
  allocAndInitHipDeviceData(a, m_a, iend); \
  allocAndInitHipDeviceData(c, m_c, iend);

#define COPY_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_c, c, iend); \
  deallocHipDeviceData(a); \
  deallocHipDeviceData(c);

__global__ void copy(Real_ptr c, Real_ptr a,
                     Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     COPY_BODY;
   }
}


void COPY::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_HIP ) {

    COPY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL((copy), dim3(grid_size), dim3(block_size), 0, 0,  c, a,
                                        iend );

    }
    stopTimer();

    COPY_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    COPY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         COPY_BODY;
       });

    }
    stopTimer();

    COPY_DATA_TEARDOWN_HIP;

  } else {
      std::cout << "\n  COPY : Unknown Hip variant id = " << vid << std::endl;
   }

}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
