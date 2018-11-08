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

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define HYDRO_2D_DATA_SETUP_HIP \
  Real_ptr za; \
  Real_ptr zb; \
  Real_ptr zm; \
  Real_ptr zp; \
  Real_ptr zq; \
  Real_ptr zr; \
  Real_ptr zu; \
  Real_ptr zv; \
  Real_ptr zz; \
\
  Real_ptr zrout; \
  Real_ptr zzout; \
\
  const Real_type s = m_s; \
  const Real_type t = m_t; \
\
  const Index_type jn = m_jn; \
  const Index_type kn = m_kn; \
\
  allocAndInitHipDeviceData(za, m_za, m_array_length); \
  allocAndInitHipDeviceData(zb, m_zb, m_array_length); \
  allocAndInitHipDeviceData(zm, m_zm, m_array_length); \
  allocAndInitHipDeviceData(zp, m_zp, m_array_length); \
  allocAndInitHipDeviceData(zq, m_zq, m_array_length); \
  allocAndInitHipDeviceData(zr, m_zr, m_array_length); \
  allocAndInitHipDeviceData(zu, m_zu, m_array_length); \
  allocAndInitHipDeviceData(zv, m_zv, m_array_length); \
  allocAndInitHipDeviceData(zz, m_zz, m_array_length); \
  allocAndInitHipDeviceData(zrout, m_zrout, m_array_length); \
  allocAndInitHipDeviceData(zzout, m_zzout, m_array_length);

#define HYDRO_2D_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_zrout, zrout, m_array_length); \
  getHipDeviceData(m_zzout, zzout, m_array_length); \
  deallocHipDeviceData(za); \
  deallocHipDeviceData(zb); \
  deallocHipDeviceData(zm); \
  deallocHipDeviceData(zp); \
  deallocHipDeviceData(zq); \
  deallocHipDeviceData(zr); \
  deallocHipDeviceData(zu); \
  deallocHipDeviceData(zv); \
  deallocHipDeviceData(zz); \
  deallocHipDeviceData(zrout); \
  deallocHipDeviceData(zzout);

#define HYDRO_2D_DATA_SETUP_HIP_RAJA \
  Real_ptr zadat; \
  Real_ptr zbdat; \
  Real_ptr zmdat; \
  Real_ptr zpdat; \
  Real_ptr zqdat; \
  Real_ptr zrdat; \
  Real_ptr zudat; \
  Real_ptr zvdat; \
  Real_ptr zzdat; \
\
  Real_ptr zroutdat; \
  Real_ptr zzoutdat; \
\
  const Real_type s = m_s; \
  const Real_type t = m_t; \
\
  const Index_type jn = m_jn; \
  const Index_type kn = m_kn; \
\
  allocAndInitHipDeviceData(zadat, m_za, m_array_length); \
  allocAndInitHipDeviceData(zbdat, m_zb, m_array_length); \
  allocAndInitHipDeviceData(zmdat, m_zm, m_array_length); \
  allocAndInitHipDeviceData(zpdat, m_zp, m_array_length); \
  allocAndInitHipDeviceData(zqdat, m_zq, m_array_length); \
  allocAndInitHipDeviceData(zrdat, m_zr, m_array_length); \
  allocAndInitHipDeviceData(zudat, m_zu, m_array_length); \
  allocAndInitHipDeviceData(zvdat, m_zv, m_array_length); \
  allocAndInitHipDeviceData(zzdat, m_zz, m_array_length); \
  allocAndInitHipDeviceData(zroutdat, m_zrout, m_array_length); \
  allocAndInitHipDeviceData(zzoutdat, m_zzout, m_array_length);

#define HYDRO_2D_DATA_TEARDOWN_HIP_RAJA \
  getHipDeviceData(m_zrout, zroutdat, m_array_length); \
  getHipDeviceData(m_zzout, zzoutdat, m_array_length); \
  deallocHipDeviceData(zadat); \
  deallocHipDeviceData(zbdat); \
  deallocHipDeviceData(zmdat); \
  deallocHipDeviceData(zpdat); \
  deallocHipDeviceData(zqdat); \
  deallocHipDeviceData(zrdat); \
  deallocHipDeviceData(zudat); \
  deallocHipDeviceData(zvdat); \
  deallocHipDeviceData(zzdat); \
  deallocHipDeviceData(zroutdat); \
  deallocHipDeviceData(zzoutdat);

__global__ void hydro_2d1(Real_ptr za, Real_ptr zb, 
                          Real_ptr zp, Real_ptr zq, Real_ptr zr, Real_ptr zm,
                          Index_type jn, Index_type kn) 
{
   Index_type k = blockIdx.y;
   Index_type j = threadIdx.x;
   if (k > 0 && k < kn-1 && j > 0 && j < jn-1) {
     HYDRO_2D_BODY1; 
   }
}

__global__ void hydro_2d2(Real_ptr zu, Real_ptr zv,
                          Real_ptr za, Real_ptr zb, Real_ptr zz, Real_ptr zr,
                          Real_type s,
                          Index_type jn, Index_type kn)
{
   Index_type k = blockIdx.y;
   Index_type j = threadIdx.x;
   if (k > 0 && k < kn-1 && j > 0 && j < jn-1) {
     HYDRO_2D_BODY2;
   }
}

__global__ void hydro_2d3(Real_ptr zrout, Real_ptr zzout,
                          Real_ptr zr, Real_ptr zu, Real_ptr zz, Real_ptr zv,
                          Real_type t,
                          Index_type jn, Index_type kn)
{
   Index_type k = blockIdx.y;
   Index_type j = threadIdx.x;
   if (k > 0 && k < kn-1 && j > 0 && j < jn-1) {
     HYDRO_2D_BODY3;
   }
}


void HYDRO_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  if ( vid == Base_HIP ) {

    HYDRO_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       dim3 nthreads_per_block(jn, 1, 1);
       dim3 nblocks(1, kn, 1);

       hipLaunchKernelGGL((hydro_2d1), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                                                  za, zb,
                                                  zp, zq, zr, zm,
                                                  jn, kn);

       hipLaunchKernelGGL((hydro_2d2), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                                                  zu, zv,
                                                  za, zb, zz, zr,
                                                  s,
                                                  jn, kn);

       hipLaunchKernelGGL((hydro_2d3), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                                                  zrout, zzout,
                                                  zr, zu, zz, zv,
                                                  t,
                                                  jn, kn);

    }
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    HYDRO_2D_DATA_SETUP_HIP_RAJA;

    HYDRO_2D_VIEWS_RAJA;

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::HipKernelAsync<
            RAJA::statement::For<0, RAJA::hip_block_exec,  // k
              RAJA::statement::For<1, RAJA::hip_thread_exec,  // j
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });

    }
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_HIP_RAJA;

  } else { 
     std::cout << "\n  HYDRO_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
