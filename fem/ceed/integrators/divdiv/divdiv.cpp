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

#include "divdiv.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "divdiv_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct DivDivOperatorInfo : public OperatorInfo
{
   DivDivContext ctx = {0};
   DivDivOperatorInfo(const mfem::FiniteElementSpace &fes, mfem::Coefficient *Q,
                      bool use_bdr = false, bool use_mf = false)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      if (!use_mf)
      {
         apply_func = ":f_apply_divdiv";
         apply_qf = &f_apply_divdiv;
      }
      else
      {
         build_func = "";
         build_qf = nullptr;
      }
      if (Q == nullptr)
      {
         ctx.coeff = 1.0;
         if (!use_mf)
         {
            build_func = ":f_build_divdiv_const";
            build_qf = &f_build_divdiv_const;
         }
         else
         {
            apply_func = ":f_apply_divdiv_mf_const";
            apply_qf = &f_apply_divdiv_mf_const;
         }
      }
      else if (mfem::ConstantCoefficient *const_coeff =
                  dynamic_cast<mfem::ConstantCoefficient *>(Q))
      {
         ctx.coeff = const_coeff->constant;
         if (!use_mf)
         {
            build_func = ":f_build_divdiv_const";
            build_qf = &f_build_divdiv_const;
         }
         else
         {
            apply_func = ":f_apply_divdiv_mf_const";
            apply_qf = &f_apply_divdiv_mf_const;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_divdiv_quad";
            build_qf = &f_build_divdiv_quad;
         }
         else
         {
            apply_func = ":f_apply_divdiv_mf_quad";
            apply_qf = &f_apply_divdiv_mf_quad;
         }
      }
      header = "/integrators/divdiv/divdiv_qf.h";
      trial_op = EvalMode::Div;
      test_op = EvalMode::Div;
      qdatasize = 1;
   }
};
#endif

PADivDivIntegrator::PADivDivIntegrator(const mfem::DivDivIntegrator &integ,
                                       const mfem::FiniteElementSpace &fes,
                                       mfem::Coefficient *Q,
                                       const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   DivDivOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFDivDivIntegrator::MFDivDivIntegrator(const mfem::DivDivIntegrator &integ,
                                       const mfem::FiniteElementSpace &fes,
                                       mfem::Coefficient *Q,
                                       const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   DivDivOperatorInfo info(fes, Q, use_bdr, true);
   Assemble(integ, info, fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
