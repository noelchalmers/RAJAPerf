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

///
/// Methods for HIP kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_HipDataUtils_HPP
#define RAJAPerf_HipDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_HIP)


#include "RAJA/policy/hip/raja_hiperrchk.hpp"


namespace rajaperf
{

/*!
 * \brief Copy given hptr (host) data to HIP device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initHipDeviceData(T& dptr, const T hptr, int len)
{
  hipErrchk( hipMemcpy( dptr, hptr,
                          len * sizeof(typename std::remove_pointer<T>::type),
                          hipMemcpyHostToDevice ) );

  incDataInitCount();
}

/*!
 * \brief Allocate HIP device data array (dptr) and copy given hptr (host)
 * data to device array.
 */
template <typename T>
void allocAndInitHipDeviceData(T& dptr, const T hptr, int len)
{
  hipErrchk( hipMalloc( (void**)&dptr,
              len * sizeof(typename std::remove_pointer<T>::type) ) );

  initHipDeviceData(dptr, hptr, len);
}

/*!
 * \brief Copy given dptr (HIP device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getHipDeviceData(T& hptr, const T dptr, int len)
{
  hipErrchk( hipMemcpy( hptr, dptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipMemcpyDeviceToHost ) );
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocHipDeviceData(T& dptr)
{
  hipErrchk( hipFree( dptr ) );
  dptr = 0;
}


}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard

