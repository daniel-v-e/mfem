// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mixedvecgrad.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "../diffusion/diffusion_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct MixedVectorGradientOperatorInfoBase : public OperatorInfo
{
   DiffusionContext ctx = {0};
   template <typename CoeffType>
   MixedVectorGradientOperatorInfoBase(const mfem::FiniteElementSpace &trial_fes,
                                       const mfem::FiniteElementSpace &test_fes,
                                       CoeffType *Q, bool use_bdr = false,
                                       bool use_mf = false)
   {
      // Reuse H(curl) quadrature functions for DiffusionIntegrator
      MFEM_VERIFY(trial_fes.GetVDim() == 1 && test_fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = trial_fes.GetMesh()->Dimension() - use_bdr;
      MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
                  "MixedVectorGradientIntegrator and MixedVectorWeakDivergenceIntegrator "
                  "require dim == 2 or dim == 3!");
      ctx.space_dim = trial_fes.GetMesh()->SpaceDimension();
      ctx.vdim = 1;
      if (!use_mf)
      {
         apply_func = ":f_apply_diff";
         apply_qf = &f_apply_diff;
      }
      else
      {
         build_func = "";
         build_qf = nullptr;
      }
      if (Q == nullptr)
      {
         ctx.coeff[0] = 1.0;
         if (!use_mf)
         {
            build_func = ":f_build_diff_const_scalar";
            build_qf = &f_build_diff_const_scalar;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_const_scalar";
            apply_qf = &f_apply_diff_mf_const_scalar;
         }
      }
      else
      {
         InitCoefficient(*Q, use_mf);
      }
      header = "/integrators/diffusion/diffusion_qf.h";
      qdatasize = (ctx.dim * (ctx.dim + 1)) / 2;
   }
   void InitCoefficient(mfem::Coefficient &Q, bool use_mf)
   {
      if (mfem::ConstantCoefficient *const_coeff =
             dynamic_cast<mfem::ConstantCoefficient *>(&Q))
      {
         ctx.coeff[0] = const_coeff->constant;
         if (!use_mf)
         {
            build_func = ":f_build_diff_const_scalar";
            build_qf = &f_build_diff_const_scalar;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_const_scalar";
            apply_qf = &f_apply_diff_mf_const_scalar;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_diff_quad_scalar";
            build_qf = &f_build_diff_quad_scalar;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_quad_scalar";
            apply_qf = &f_apply_diff_mf_quad_scalar;
         }
      }
   }
   void InitCoefficient(mfem::VectorCoefficient &VQ, bool use_mf)
   {
      if (mfem::VectorConstantCoefficient *const_coeff =
             dynamic_cast<mfem::VectorConstantCoefficient *>(&VQ))
      {
         const int vdim = VQ.GetVDim();
         MFEM_VERIFY(vdim <= LIBCEED_DIFF_COEFF_COMP_MAX,
                     "VectorCoefficient dimension exceeds context storage!");
         const mfem::Vector &val = const_coeff->GetVec();
         for (int i = 0; i < vdim; i++)
         {
            ctx.coeff[i] = val[i];
         }
         if (!use_mf)
         {
            build_func = ":f_build_diff_const_vector";
            build_qf = &f_build_diff_const_vector;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_const_vector";
            apply_qf = &f_apply_diff_mf_const_vector;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_diff_quad_vector";
            build_qf = &f_build_diff_quad_vector;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_quad_vector";
            apply_qf = &f_apply_diff_mf_quad_vector;
         }
      }
   }
   void InitCoefficient(mfem::MatrixCoefficient &MQ, bool use_mf)
   {
      // Assumes matrix coefficient is symmetric
      if (mfem::MatrixConstantCoefficient *const_coeff =
             dynamic_cast<mfem::MatrixConstantCoefficient *>(&MQ))
      {
         const int vdim = MQ.GetVDim();
         MFEM_VERIFY((vdim * (vdim + 1)) / 2 <= LIBCEED_DIFF_COEFF_COMP_MAX,
                     "MatrixCoefficient dimensions exceed context storage!");
         const mfem::DenseMatrix &val = const_coeff->GetMatrix();
         for (int j = 0; j < vdim; j++)
         {
            for (int i = j; i < vdim; i++)
            {
               const int idx = (j * vdim) - (((j - 1) * j) / 2) + i - j;
               ctx.coeff[idx] = val(i, j);
            }
         }
         if (!use_mf)
         {
            build_func = ":f_build_diff_const_matrix";
            build_qf = &f_build_diff_const_matrix;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_const_matrix";
            apply_qf = &f_apply_diff_mf_const_matrix;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_diff_quad_matrix";
            build_qf = &f_build_diff_quad_matrix;
         }
         else
         {
            apply_func = ":f_apply_diff_mf_quad_matrix";
            apply_qf = &f_apply_diff_mf_quad_matrix;
         }
      }
   }
};

struct MixedVectorGradientOperatorInfo :
   public MixedVectorGradientOperatorInfoBase
{
   template <typename CoeffType>
   MixedVectorGradientOperatorInfo(const mfem::FiniteElementSpace &trial_fes,
                                   const mfem::FiniteElementSpace &test_fes,
                                   CoeffType *Q, bool use_bdr = false,
                                   bool use_mf = false)
      : MixedVectorGradientOperatorInfoBase(trial_fes, test_fes, Q, use_bdr, use_mf)
   {
      MFEM_VERIFY(
         (trial_fes.FEColl()->GetDerivMapType(ctx.dim) == mfem::FiniteElement::H_CURL &&
          test_fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_CURL),
         "libCEED interface for MixedVectorGradientIntegrator requires "
         "H^1 domain and H(curl) range FE spaces!");
      trial_op = EvalMode::Grad;
      test_op = EvalMode::Interp;
   }
};

struct MixedVectorWeakDivergenceOperatorInfo :
   public MixedVectorGradientOperatorInfoBase
{
   template <typename CoeffType>
   MixedVectorWeakDivergenceOperatorInfo(const mfem::FiniteElementSpace &trial_fes,
                                         const mfem::FiniteElementSpace &test_fes,
                                         CoeffType *Q, bool use_bdr = false,
                                         bool use_mf = false)
      : MixedVectorGradientOperatorInfoBase(trial_fes, test_fes, Q, use_bdr, use_mf)
   {
      MFEM_VERIFY(
         (trial_fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_CURL &&
          test_fes.FEColl()->GetDerivMapType(ctx.dim) == mfem::FiniteElement::H_CURL),
         "libCEED interface for MixedVectorWeakDivergenceIntegrator requires "
         "H(curl) domain and H^1 range FE spaces!");
      trial_op = EvalMode::Interp;
      test_op = EvalMode::Grad;
      for (int i = 0; i < LIBCEED_DIFF_COEFF_COMP_MAX; i++)
      {
         ctx.coeff[i] *= -1.0;
      }
   }
};
#endif

template <typename CoeffType>
PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, true);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

namespace
{

#ifdef MFEM_USE_CEED
mfem::Coefficient *NegativeCoeff(mfem::Coefficient &Q)
{
   return (dynamic_cast<mfem::ConstantCoefficient *>(&Q) != nullptr) ?
          nullptr : new mfem::ProductCoefficient(-1.0, Q);
}

mfem::VectorCoefficient *NegativeCoeff(mfem::VectorCoefficient &Q)
{
   return (dynamic_cast<mfem::VectorConstantCoefficient *>(&Q) != nullptr) ?
          nullptr : new mfem::ScalarVectorProductCoefficient(-1.0, Q);
}

mfem::MatrixCoefficient *NegativeCoeff(mfem::MatrixCoefficient &Q)
{
   return (dynamic_cast<mfem::MatrixConstantCoefficient *>(&Q) != nullptr) ?
          nullptr : new mfem::ScalarMatrixProductCoefficient(-1.0, Q);
}
#endif

} // namespace

template <typename CoeffType>
PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorWeakDivergenceOperatorInfo info(trial_fes, test_fes, Q, use_bdr);
   if (Q)
   {
      // Does not inherit ownership of old Q
      auto *nQ = NegativeCoeff(*Q);
      Assemble(integ, info, trial_fes, test_fes, nQ, use_bdr);
      delete nQ;
   }
   else
   {
      Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
   }
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorWeakDivergenceOperatorInfo info(trial_fes, test_fes, Q, use_bdr,
                                              true);
   if (Q)
   {
      // Does not inherit ownership of old Q
      auto *nQ = NegativeCoeff(*Q);
      Assemble(integ, info, trial_fes, test_fes, nQ, use_bdr, true);
      delete nQ;
   }
   else
   {
      Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
   }
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

// @cond DOXYGEN_SKIP

template PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

template PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &,
   const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &,
   const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &,
   const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

template MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

template MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &,
   const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &,
   const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &,
   const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

// @endcond

} // namespace ceed

} // namespace mfem
