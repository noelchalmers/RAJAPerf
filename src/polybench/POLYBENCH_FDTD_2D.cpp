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
#include "common/DataUtils.hpp"

#include <iostream>
#include <cstring>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_FDTD_2D_DATA_SETUP_CPU \
  const Index_type nx = m_nx; \
  const Index_type ny = m_ny; \
  const Index_type tsteps = m_tsteps; \
\
  ResReal_ptr fict = m_fict; \
  ResReal_ptr ex = m_ex; \
  ResReal_ptr ey = m_ey; \
  ResReal_ptr hz = m_hz; 

POLYBENCH_FDTD_2D::POLYBENCH_FDTD_2D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_FDTD_2D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps;
  switch(lsizespec) {
    case Mini:
      m_nx=20; m_ny=30; m_tsteps=20; 
      run_reps = 10000;
      break;
    case Small:
      m_nx=60; m_ny=80; m_tsteps=40; 
      run_reps = 500;
      break;
    case Medium:
      m_nx=200; m_ny=240; m_tsteps=100; 
      run_reps = 200;
      break;
    case Large:
      m_nx=800; m_ny=1000; m_tsteps=500; 
      run_reps = 1;
      break;
    case Extralarge:
      m_nx=2000; m_ny=2600; m_tsteps=1000; 
      run_reps = 1;
      break;
    default:
      m_nx=800; m_ny=1000; m_tsteps=60; 
      run_reps = 10;
      break;
  }
  setDefaultSize( m_tsteps * (m_ny + 3 * m_nx*m_ny) );
  setDefaultReps(run_reps);
}

POLYBENCH_FDTD_2D::~POLYBENCH_FDTD_2D() 
{

}

void POLYBENCH_FDTD_2D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_hz, m_nx * m_ny, 0.0, vid);
  allocAndInitData(m_ex, m_nx * m_ny, vid);
  allocAndInitData(m_ey, m_nx * m_ny, vid);
  allocAndInitData(m_fict, m_tsteps, vid);
}

void POLYBENCH_FDTD_2D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_FDTD_2D_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

          for (Index_type j = 0; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY1;
          }
          for (Index_type i = 1; i < nx; i++) {
            for (Index_type j = 0; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY2;
            }
          }
          for (Index_type i = 0; i < nx; i++) {
            for (Index_type j = 1; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY3;
            }
          }
          for (Index_type i = 0; i < nx - 1; i++) {
            for (Index_type j = 0; j < ny - 1; j++) {
              POLYBENCH_FDTD_2D_BODY4;
            }
          }

        }  // tstep loop

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      POLYBENCH_FDTD_2D_DATA_SETUP_CPU;

      POLYBENCH_FDTD_2D_VIEWS_RAJA;

      using EXEC_POL1 = RAJA::loop_exec;

      using EXEC_POL234 =  
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

          RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny), 
           [=](Index_type j) {
             POLYBENCH_FDTD_2D_BODY1_RAJA;
          });

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                             RAJA::RangeSegment{0, ny}),
            [=](Index_type i, Index_type j) {
              POLYBENCH_FDTD_2D_BODY2_RAJA;
            }
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                             RAJA::RangeSegment{1, ny}),
            [=](Index_type i, Index_type j) {
              POLYBENCH_FDTD_2D_BODY3_RAJA;
            }
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                             RAJA::RangeSegment{0, ny-1}),
            [=](Index_type i, Index_type j) {
              POLYBENCH_FDTD_2D_BODY4_RAJA;
            }
          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      POLYBENCH_FDTD_2D_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type j = 0; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY1;
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < nx; i++) {
            for (Index_type j = 0; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY2;
            }
          }
          #pragma omp parallel for
          for (Index_type i = 0; i < nx; i++) {
            for (Index_type j = 1; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY3;
            }
          }
          #pragma omp parallel for
          for (Index_type i = 0; i < nx - 1; i++) {
            for (Index_type j = 0; j < ny - 1; j++) {
              POLYBENCH_FDTD_2D_BODY4;
            }
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_FDTD_2D_DATA_SETUP_CPU;

      POLYBENCH_FDTD_2D_VIEWS_RAJA;

      using EXEC_POL1 = RAJA::omp_parallel_for_exec;

      using EXEC_POL234 =  
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny),
           [=](Index_type j) {
             POLYBENCH_FDTD_2D_BODY1_RAJA;
          });

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                             RAJA::RangeSegment{0, ny}),
            [=](Index_type i, Index_type j) {
              POLYBENCH_FDTD_2D_BODY2_RAJA;
            }
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                             RAJA::RangeSegment{1, ny}),
            [=](Index_type i, Index_type j) {
              POLYBENCH_FDTD_2D_BODY3_RAJA;
            }
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                             RAJA::RangeSegment{0, ny-1}),
            [=](Index_type i, Index_type j) {
              POLYBENCH_FDTD_2D_BODY4_RAJA;
            }
          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_HIP)
    case Base_HIP :
    case RAJA_HIP :
    {
      runHipVariant(vid);
      break;
    }
#endif
    
    default : {
      std::cout << "\nPOLYBENCH_FDTD_2D  Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_FDTD_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_hz, m_nx * m_ny);
}

void POLYBENCH_FDTD_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_fict);
  deallocData(m_ex);
  deallocData(m_ey);
  deallocData(m_hz);
}

} // end namespace polybench
} // end namespace rajaperf
