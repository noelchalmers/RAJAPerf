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
/// POLYBENCH_ATAX kernel reference implementation:
///
/// for (int i = 0; i < N; i++) {
///   y[i] = 0;
///   tmp[i] = 0;
///   for (int j = 0; j < N; j++) {
///     tmp[i] += A[i][j] * x[j];
///   }
/// }
/// for (int j = 0; j < N; j++) {
///   for (int i = 0; i < N; i++) {
///     y[j] += A[i][j] * tmp[i];
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_ATAX_HPP
#define RAJAPerf_POLYBENCH_ATAX_HPP

#define POLYBENCH_ATAX_BODY1 \
  y[i] = 0.0; \
  tmp[i] = 0.0;

#define POLYBENCH_ATAX_BODY2 \
  tmp[i] += A[j + i*N] * x[j];

#define POLYBENCH_ATAX_BODY3 \
  y[j] += A[j + i*N] * tmp[i];


#define POLYBENCH_ATAX_BODY1_RAJA \
  yview(i) = 0.0; \
  tmpview(i) = 0.0;

#define POLYBENCH_ATAX_BODY2_RAJA \
  tmpview(i) += Aview(i, j) * xview(j); 

#define POLYBENCH_ATAX_BODY3_RAJA \
  yview(j) += Aview(i, j) * tmpview(i); 

#define POLYBENCH_ATAX_VIEWS_RAJA \
  using VIEW_1 = RAJA::View<Real_type, \
                            RAJA::Layout<1, Index_type, 0>>; \
\
  using VIEW_2 = RAJA::View<Real_type, \
                            RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_1 tmpview(tmp, RAJA::Layout<1>(N)); \
  VIEW_1 xview(x, RAJA::Layout<1>(N)); \
  VIEW_1 yview(y, RAJA::Layout<1>(N)); \
  VIEW_2 Aview(A, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_ATAX : public KernelBase
{
public:

  POLYBENCH_ATAX(const RunParams& params);

  ~POLYBENCH_ATAX();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_N;
  Real_ptr m_tmp;
  Real_ptr m_y;
  Real_ptr m_x;
  Real_ptr m_A;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
