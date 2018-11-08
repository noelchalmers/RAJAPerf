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
/// POLYBENCH_FDTD_2D kernel reference implementation:
///
/// for (t = 0; t < TSTEPS; t++)
/// {
///   for (j = 0; j < ny; j++) {
///	ey[0][j] = fict[t];
///   }
///   for (i = 1; i < nx; i++) {
///	for (j = 0; j < ny; j++) {
///	  ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
///     }
///   } 
///   for (i = 0; i < nx; i++) {
///	for (j = 1; j < ny; j++) {
///	  ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
///     }
///   } 
///   for (i = 0; i < nx - 1; i++) {
///	for (j = 0; j < ny - 1; j++) {
///	  hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] +
///                                  ey[i+1][j] - ey[i][j]);
///     }
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_FDTD_2D_HPP
#define RAJAPerf_POLYBENCH_FDTD_2D_HPP

#define POLYBENCH_FDTD_2D_BODY1 \
  ey[j + 0*ny] = fict[t];

#define POLYBENCH_FDTD_2D_BODY2 \
  ey[j + i*ny] = ey[j + i*ny] - 0.5*(hz[j + i*ny] - hz[j + (i-1)*ny]); 

#define POLYBENCH_FDTD_2D_BODY3 \
  ex[j + i*ny] = ex[j + i*ny] - 0.5*(hz[j + i*ny] - hz[j-1 + i*ny]); 

#define POLYBENCH_FDTD_2D_BODY4 \
  hz[j + i*ny] = hz[j + i*ny] - 0.7*(ex[j+1 + i*ny] - ex[j + i*ny] + \
                                     ey[j + (i+1)*ny] - ey[j + i*ny]); 


#define POLYBENCH_FDTD_2D_BODY1_RAJA \
  eyview(0, j) = fict[t];

#define POLYBENCH_FDTD_2D_BODY2_RAJA \
  eyview(i, j) = eyview(i, j) - 0.5*(hzview(i, j) - hzview(i-1, j));

#define POLYBENCH_FDTD_2D_BODY3_RAJA \
  exview(i, j) = exview(i, j) - 0.5*(hzview(i, j) - hzview(i, j-1));

#define POLYBENCH_FDTD_2D_BODY4_RAJA \
  hzview(i, j) = hzview(i, j) - 0.7*(exview(i, j+1) - exview(i, j) + \
                                     eyview(i+1, j) - eyview(i, j));

#define POLYBENCH_FDTD_2D_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE exview(ex, RAJA::Layout<2>(nx, ny)); \
  VIEW_TYPE eyview(ey, RAJA::Layout<2>(nx, ny)); \
  VIEW_TYPE hzview(hz, RAJA::Layout<2>(nx, ny));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_FDTD_2D : public KernelBase
{
public:

  POLYBENCH_FDTD_2D(const RunParams& params);

  ~POLYBENCH_FDTD_2D();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_nx;
  Index_type m_ny;
  Index_type m_tsteps;

  Real_ptr m_fict;
  Real_ptr m_ex;
  Real_ptr m_ey;
  Real_ptr m_hz;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
