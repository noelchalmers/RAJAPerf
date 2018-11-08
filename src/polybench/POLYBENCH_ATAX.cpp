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
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_ATAX_DATA_SETUP_CPU \
  ResReal_ptr tmp = m_tmp; \
  ResReal_ptr y = m_y; \
  ResReal_ptr x = m_x; \
  ResReal_ptr A = m_A;

  
POLYBENCH_ATAX::POLYBENCH_ATAX(const RunParams& params)
  : KernelBase(rajaperf::Polybench_ATAX, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=42;
      run_reps = 10000;
      break;
    case Small:
      m_N=124;
      run_reps = 1000;
      break;
    case Medium:
      m_N=410;
      run_reps = 100;
      break;
    case Large:
      m_N=2100;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=2200;
      run_reps = 1;
      break;
    default:
      m_N=2100;
      run_reps = 100;
      break;
  }

  setDefaultSize( m_N + m_N*2*m_N );
  setDefaultReps(run_reps);
}

POLYBENCH_ATAX::~POLYBENCH_ATAX() 
{

}

void POLYBENCH_ATAX::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_tmp, m_N, vid);
  allocAndInitData(m_x, m_N, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitDataConst(m_y, m_N, 0.0, vid);
}

void POLYBENCH_ATAX::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();
  const Index_type N = m_N;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_ATAX_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < N; ++i ) { 
          POLYBENCH_ATAX_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_ATAX_BODY2;
          }
        }

        for (Index_type i = 0; i < N; ++i ) { 
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_ATAX_BODY3;
          }
        }

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      POLYBENCH_ATAX_DATA_SETUP_CPU;

      POLYBENCH_ATAX_VIEWS_RAJA;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >,
          RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::For<0, RAJA::loop_exec,
              RAJA::statement::Lambda<2>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N}),
          [=](Index_type i, Index_type /* j */) {
            POLYBENCH_ATAX_BODY1_RAJA;
          },
          [=](Index_type i, Index_type j) {
            POLYBENCH_ATAX_BODY2_RAJA;
          },
          [=](Index_type i, Index_type j) {
            POLYBENCH_ATAX_BODY3_RAJA;
          }
        );

      }
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      POLYBENCH_ATAX_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_ATAX_BODY2;
          }
        }

        #pragma omp parallel for
        for (Index_type j = 0; j < N; ++j ) {
          for (Index_type i = 0; i < N; ++i ) {
            POLYBENCH_ATAX_BODY3;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_ATAX_DATA_SETUP_CPU;

      POLYBENCH_ATAX_VIEWS_RAJA;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >,
          RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<0, RAJA::loop_exec,
              RAJA::statement::Lambda<2>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N}),
          [=](Index_type i, Index_type /* j */) {
            POLYBENCH_ATAX_BODY1_RAJA;
          },
          [=](Index_type i, Index_type j) {
            POLYBENCH_ATAX_BODY2_RAJA;
          },
          [=](Index_type i, Index_type j) {
            POLYBENCH_ATAX_BODY3_RAJA;
          }
        );

      }
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
      std::cout << "\n  POLYBENCH_ATAX : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_ATAX::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, m_N);
}

void POLYBENCH_ATAX::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_tmp);
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_A);
}

} // end namespace polybench
} // end namespace rajaperf
