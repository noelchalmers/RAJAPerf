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

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_JACOBI_1D_DATA_SETUP_CPU \
  ResReal_ptr A = m_Ainit; \
  ResReal_ptr B = m_Binit;

#define POLYBENCH_JACOBI_1D_DATA_RESET_CPU \
  m_Ainit = m_A; \
  m_Binit = m_B; \
  m_A = A; \
  m_B = B; 
  
POLYBENCH_JACOBI_1D::POLYBENCH_JACOBI_1D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_JACOBI_1D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=300;
      m_tsteps=20;
      run_reps = 10000;
      break;
    case Small:
      m_N=1200;
      m_tsteps=100;
      run_reps = 1000;
      break;
    case Medium:
      m_N=4000;
      m_tsteps=100;
      run_reps = 100;
      break;
    case Large:
      m_N=200000;
      m_tsteps=50;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=2000000;
      m_tsteps=10;
      run_reps = 20;
      break;
    default:
      m_N=4000000;
      m_tsteps=10;
      run_reps = 10;
      break;
  }

  setDefaultSize( m_tsteps * 2 * m_N );
  setDefaultReps(run_reps);
}

POLYBENCH_JACOBI_1D::~POLYBENCH_JACOBI_1D() 
{

}

void POLYBENCH_JACOBI_1D::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N, vid);
  allocAndInitData(m_Binit, m_N, vid);
  allocAndInitDataConst(m_A, m_N, 0.0, vid);
  allocAndInitDataConst(m_B, m_N, 0.0, vid);
}

void POLYBENCH_JACOBI_1D::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();
  const Index_type N = m_N;
  const Index_type tsteps = m_tsteps;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_JACOBI_1D_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 
          for (Index_type i = 1; i < N-1; ++i ) { 
            POLYBENCH_JACOBI_1D_BODY1;
          }
          for (Index_type i = 1; i < N-1; ++i ) { 
            POLYBENCH_JACOBI_1D_BODY2;
          }
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }


#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      POLYBENCH_JACOBI_1D_DATA_SETUP_CPU;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>
          >,
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<1>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1}),
            [=](Index_type i) {
              POLYBENCH_JACOBI_1D_BODY1;
            },
            [=](Index_type i) {
              POLYBENCH_JACOBI_1D_BODY2;
            }
          );
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      POLYBENCH_JACOBI_1D_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            POLYBENCH_JACOBI_1D_BODY1;
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            POLYBENCH_JACOBI_1D_BODY2;
          }
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_JACOBI_1D_DATA_SETUP_CPU;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0>
          >,
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<1>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1}),
            [=](Index_type i) {
              POLYBENCH_JACOBI_1D_BODY1;
            },
            [=](Index_type i) {
              POLYBENCH_JACOBI_1D_BODY2;
            }
          );
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

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
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_JACOBI_1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_A, m_N);
  checksum[vid] += calcChecksum(m_B, m_N);
}

void POLYBENCH_JACOBI_1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_Ainit);
  deallocData(m_Binit);
}

} // end namespace polybench
} // end namespace rajaperf
