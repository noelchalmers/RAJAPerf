//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
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
/// POLYBENCH_GESUMMV kernel reference implementation:
///
/// for (Index_type i = 0; i < N; i++) {
///     tmp[i] = 0.0;
///     y[i] = 0.0;
///   for (Index_type j = 0; j < N; j++) {
///     tmp[i] += A[i][j] * x[j];
///     y[i] += B[i][j] * x[j];
///   }
///   y[i] = alpha * tmp[i] + beta * y[i];
/// }


#ifndef RAJAPerf_POLYBENCH_GESUMMV_HPP
#define RAJAPerf_POLYBENCH_GESUMMV_HPP

#define POLYBENCH_GESUMMV_BODY1 \
  Real_type tmpdot = 0.0; \
  Real_type ydot = 0.0;

#define POLYBENCH_GESUMMV_BODY2 \
  tmpdot += A[j + i*N] * x[j]; \
  ydot += B[j + i*N] * x[j];

#define POLYBENCH_GESUMMV_BODY3 \
  y[i] = alpha * tmpdot + beta * ydot;


#define POLYBENCH_GESUMMV_BODY1_RAJA \
  tmpdot = 0.0; \
  ydot = 0.0;

#define POLYBENCH_GESUMMV_BODY2_RAJA \
  tmpdot += Aview(i, j) * xview(j); \
  ydot += Bview(i, j) * xview(j);

#define POLYBENCH_GESUMMV_BODY3_RAJA \
  yview(i) = alpha * tmpdot + beta * ydot;


#define POLYBENCH_GESUMMV_VIEWS_RAJA \
  using VIEW1_TYPE = RAJA::View<Real_type, \
                                 RAJA::Layout<1, Index_type, 0>>; \
  using VIEW2_TYPE = RAJA::View<Real_type, \
                                 RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW1_TYPE xview(x, RAJA::Layout<1>(N)); \
  VIEW1_TYPE yview(y, RAJA::Layout<1>(N)); \
  VIEW2_TYPE Aview(A, RAJA::Layout<2>(N, N)); \
  VIEW2_TYPE Bview(B, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_GESUMMV : public KernelBase
{
public:

  POLYBENCH_GESUMMV(const RunParams& params);

  ~POLYBENCH_GESUMMV();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_N;

  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_A;
  Real_ptr m_B;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard