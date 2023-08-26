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

#include "vecfemass.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "vecfemass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct VectorFEMassOperatorInfo : public OperatorInfo
{
   VectorFEMassContext ctx = {0};
   template <typename CoeffType>
   VectorFEMassOperatorInfo(const mfem::FiniteElementSpace &fes, CoeffType *Q,
                            bool use_bdr = false, bool use_mf = false)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      bool is_hdiv = (fes.FEColl()->GetMapType(ctx.dim) ==
                      mfem::FiniteElement::H_DIV);
      MFEM_VERIFY(is_hdiv ||
                  fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_CURL,
                  "VectorFEMassIntegrator requires H(div) or H(curl) FE space!");
      if (!use_mf)
      {
         apply_func = ":f_apply_vecfemass";
         apply_qf = &f_apply_vecfemass;
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
            build_func = is_hdiv ? ":f_build_hdivmass_const_scalar" :
                         ":f_build_hcurlmass_const_scalar";
            build_qf = is_hdiv ? &f_build_hdivmass_const_scalar :
                       &f_build_hcurlmass_const_scalar;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_const_scalar" :
                         ":f_apply_hcurlmass_mf_const_scalar";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_const_scalar :
                       &f_apply_hcurlmass_mf_const_scalar;
         }
      }
      else
      {
         InitCoefficient(*Q, is_hdiv, use_mf);
      }
      header = "/integrators/vecfemass/vecfemass_qf.h";
      trial_op = EvalMode::Interp;
      test_op = EvalMode::Interp;
      qdatasize = (ctx.dim * (ctx.dim + 1)) / 2;
   }
   void InitCoefficient(mfem::Coefficient &Q, bool is_hdiv, bool use_mf)
   {
      if (mfem::ConstantCoefficient *const_coeff =
             dynamic_cast<mfem::ConstantCoefficient *>(&Q))
      {
         ctx.coeff[0] = const_coeff->constant;
         if (!use_mf)
         {
            build_func = is_hdiv ? ":f_build_hdivmass_const_scalar" :
                         ":f_build_hcurlmass_const_scalar";
            build_qf = is_hdiv ? &f_build_hdivmass_const_scalar :
                       &f_build_hcurlmass_const_scalar;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_const_scalar" :
                         ":f_apply_hcurlmass_mf_const_scalar";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_const_scalar :
                       &f_apply_hcurlmass_mf_const_scalar;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = is_hdiv ? ":f_build_hdivmass_quad_scalar" :
                         ":f_build_hcurlmass_quad_scalar";
            build_qf = is_hdiv ? &f_build_hdivmass_quad_scalar :
                       &f_build_hcurlmass_quad_scalar;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_quad_scalar" :
                         ":f_apply_hcurlmass_mf_quad_scalar";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_quad_scalar :
                       &f_apply_hcurlmass_mf_quad_scalar;
         }
      }
   }
   void InitCoefficient(mfem::VectorCoefficient &VQ, bool is_hdiv, bool use_mf)
   {
      if (mfem::VectorConstantCoefficient *const_coeff =
             dynamic_cast<mfem::VectorConstantCoefficient *>(&VQ))
      {
         const int vdim = VQ.GetVDim();
         MFEM_VERIFY(vdim <= LIBCEED_VECFEMASS_COEFF_COMP_MAX,
                     "VectorCoefficient dimension exceeds context storage!");
         const mfem::Vector &val = const_coeff->GetVec();
         for (int i = 0; i < vdim; i++)
         {
            ctx.coeff[i] = val[i];
         }
         if (!use_mf)
         {
            build_func = is_hdiv ? ":f_build_hdivmass_const_vector" :
                         ":f_build_hcurlmass_const_vector";
            build_qf = is_hdiv ? &f_build_hdivmass_const_vector :
                       &f_build_hcurlmass_const_vector;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_const_vector" :
                         ":f_apply_hcurlmass_mf_const_vector";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_const_vector :
                       &f_apply_hcurlmass_mf_const_vector;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = is_hdiv ? ":f_build_hdivmass_quad_vector" :
                         ":f_build_hcurlmass_quad_vector";
            build_qf = is_hdiv ? &f_build_hdivmass_quad_vector :
                       &f_build_hcurlmass_quad_vector;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_quad_vector" :
                         ":f_apply_hcurlmass_mf_quad_vector";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_quad_vector :
                       &f_apply_hcurlmass_mf_quad_vector;
         }
      }
   }
   void InitCoefficient(mfem::MatrixCoefficient &MQ, bool is_hdiv, bool use_mf)
   {
      // Assumes matrix coefficient is symmetric
      if (mfem::MatrixConstantCoefficient *const_coeff =
             dynamic_cast<mfem::MatrixConstantCoefficient *>(&MQ))
      {
         const int vdim = MQ.GetVDim();
         MFEM_VERIFY((vdim * (vdim + 1)) / 2 <= LIBCEED_VECFEMASS_COEFF_COMP_MAX,
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
            build_func = is_hdiv ? ":f_build_hdivmass_const_matrix" :
                         ":f_build_hcurlmass_const_matrix";
            build_qf = is_hdiv ? &f_build_hdivmass_const_matrix :
                       &f_build_hcurlmass_const_matrix;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_const_matrix" :
                         ":f_apply_hcurlmass_mf_const_matrix";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_const_matrix :
                       &f_apply_hcurlmass_mf_const_matrix;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = is_hdiv ? ":f_build_hdivmass_quad_matrix" :
                         ":f_build_hcurlmass_quad_matrix";
            build_qf = is_hdiv ? &f_build_hdivmass_quad_matrix :
                       &f_build_hcurlmass_quad_matrix;
         }
         else
         {
            apply_func = is_hdiv ? ":f_apply_hdivmass_mf_quad_matrix" :
                         ":f_apply_hcurlmass_mf_quad_matrix";
            apply_qf = is_hdiv ? &f_apply_hdivmass_mf_quad_matrix :
                       &f_apply_hcurlmass_mf_quad_matrix;
         }
      }
   }
};
#endif

template <typename CoeffType>
PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   VectorFEMassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   VectorFEMassOperatorInfo info(fes, Q, use_bdr, true);
   Assemble(integ, info, fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

// @cond DOXYGEN_SKIP

template PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

template MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

// @endcond

} // namespace ceed

} // namespace mfem
