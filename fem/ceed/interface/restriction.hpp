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

#ifndef MFEM_LIBCEED_RESTR
#define MFEM_LIBCEED_RESTR

#include "../../fespace.hpp"
#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

/** @brief Initialize a CeedElemRestriction based on an
    mfem::FiniteElementSpace @a fes and an optional list of @a nelem elements
    of indices @a indices.

    @param[in] fes The finite element space.
    @param[in] use_bdr Create the restriction for boundary elements.
    @param[in] is_interp Create the restriction for an interpolation operator
                         (sometimes requires special treatment).
    @param[in] is_range Create the restriction for the range space of an
                        operator (sometimes requires special treatment).
    @param[in] nelem The number of elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`. If `indices == nullptr`, assumes
                       that the `FiniteElementSpace` is not mixed.
    @param[in] ceed The Ceed object.
    @param[out] restr The `CeedElemRestriction` to initialize. */
void InitRestriction(const FiniteElementSpace &fes,
                     bool use_bdr,
                     bool is_interp,
                     bool is_range,
                     int nelem,
                     const int *indices,
                     Ceed ceed,
                     CeedElemRestriction *restr);

inline void InitRestriction(const FiniteElementSpace &fes,
                            bool use_bdr,
                            int nelem,
                            const int *indices,
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   InitRestriction(fes, use_bdr, false, false, nelem, indices, ceed, restr);
}

inline void InitRestriction(const FiniteElementSpace &fes,
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   InitRestriction(fes, false, false, false, fes.GetNE(), nullptr, ceed, restr);
}

/** @brief Initialize a strided CeedElemRestriction.

    @param[in] fes Input finite element space.
    @param[in] nelem is the number of elements.
    @param[in] nqpts is the total number of quadrature points.
    @param[in] qdatasize is the number of data per quadrature point.
    @param[in] strides Array for strides between [nodes, components, elements].
                       Data for node i, component j, element k can be found in
                       the L-vector at index i*strides[0] + j*strides[1] +
                       k*strides[2]. CEED_STRIDES_BACKEND may be used with
                       vectors created by a Ceed backend.
    @param[in] ceed The Ceed object.
    @param[out] restr The `CeedElemRestriction` to initialize. */
void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem,
                            CeedInt nqpts,
                            CeedInt qdatasize,
                            const CeedInt strides[3],
                            Ceed ceed,
                            CeedElemRestriction *restr);

#endif

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_RESTR
