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

#include "mfem.hpp"
#include "unit_tests.hpp"

namespace mfem
{

constexpr double EPS = 1e-10;

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh.
//            (note: permutations of the values in the diagonal are expected)
TEST_CASE("NCMesh PA diagonal", "[NCMesh]")
{
   SECTION("Quad mesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian2D(
                        ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      int dim = 2;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         FiniteElementSpace fes(&mesh, &fec);
         FiniteElementSpace nc_fes(&nc_mesh, &fec);

         BilinearForm a(&fes);
         BilinearForm nc_a(&nc_fes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(fes.GetTrueVSize());
         Vector nc_diag(nc_fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         double error = fabs(diag.Norml2() - nc_diag.Norml2());
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
      }
   }

   SECTION("Hexa mesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian3D(
                     ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian3D(
                        ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      int dim = 3;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         FiniteElementSpace fes(&mesh, &fec);
         FiniteElementSpace nc_fes(&nc_mesh, &fec);

         BilinearForm a(&fes);
         BilinearForm nc_a(&nc_fes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(fes.GetTrueVSize());
         Vector nc_diag(nc_fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         double error = fabs(diag.Sum() - nc_diag.Sum());
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
      }
   }

} // test case


TEST_CASE("NCMesh 3D Refined Volume", "[NCMesh]")
{
   auto mesh_fname = GENERATE("../../data/ref-tetrahedron.mesh",
                              "../../data/ref-cube.mesh",
                              "../../data/ref-prism.mesh",
                              "../../data/ref-pyramid.mesh"
                             );

   auto ref_type = GENERATE(Refinement::X,
                            Refinement::Y,
                            Refinement::Z,
                            Refinement::XY,
                            Refinement::XZ,
                            Refinement::YZ,
                            Refinement::XYZ);

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   double original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].ref_type = ref_type; ref[0].index = 0;

   mesh.GeneralRefinement(ref, 1);
   double summed_volume = 0.0;
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      summed_volume += mesh.GetElementVolume(i);
   }
   REQUIRE(summed_volume == MFEM_Approx(original_volume));
} // test case


TEST_CASE("NCMesh 3D Derefined Volume", "[NCMesh]")
{
   auto mesh_fname = GENERATE("../../data/ref-tetrahedron.mesh",
                              "../../data/ref-cube.mesh",
                              "../../data/ref-prism.mesh",
                              "../../data/ref-pyramid.mesh"
                             );

   auto ref_type = GENERATE(Refinement::XYZ);

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   double original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].ref_type = ref_type; ref[0].index = 0;

   mesh.GeneralRefinement(ref, 1);

   Array<double> elem_error(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      elem_error[i] = 0.0;
   }
   mesh.DerefineByError(elem_error, 1.0);

   double derefined_volume = mesh.GetElementVolume(0);
   REQUIRE(derefined_volume == MFEM_Approx(original_volume));
} // test case


int CheckPoisson(Mesh &mesh, int order, int disabled_boundary = -1)
{
   constexpr int dim = 3;

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction sol(&fes);

   ConstantCoefficient one(1.0);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Add in essential boundary conditions
   Array<int> ess_tdof_list;
   REQUIRE(mesh.bdr_attributes.Max() > 0);

   // Mark all boundaries essential
   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   if (disabled_boundary >= 0)
   {
      bdr_attr_is_ess[mesh.bdr_attributes.Find(disabled_boundary)] = 0;
   }

   fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   REQUIRE(ess_tdof_list.Size() > 0);

   ConstantCoefficient zero(0.0);
   sol.ProjectCoefficient(zero);
   Vector B, X;
   OperatorPtr A;
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

   // Solve the system
   CG(*A, B, X, 2, 1000, 1e-20, 0.0);

   // Recover the solution
   a.RecoverFEMSolution(X, b, sol);

   // Check that X solves the system A X = B.
   A->AddMult(X, B, -1.0);
   auto residual_norm = B.Norml2();
   bool satisfy_system = residual_norm < 1e-10;
   CAPTURE(residual_norm);
   CHECK(satisfy_system);

   bool satisfy_bc = true;
   Vector tvec;
   sol.GetTrueDofs(tvec);
   for (auto dof : ess_tdof_list)
   {
      if (tvec[dof] != 0.0)
      {
         satisfy_bc = false;
         break;
      }
   }
   CHECK(satisfy_bc);
   return ess_tdof_list.Size();
};

#ifdef MFEM_USE_MPI

void CheckPoisson(ParMesh &pmesh, int order, int disabled_boundary = -1)
{
   constexpr int dim = 3;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   ParGridFunction sol(&pfes);

   ConstantCoefficient one(1.0);
   ParBilinearForm a(&pfes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   ParLinearForm b(&pfes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Add in essential boundary conditions
   Array<int> ess_tdof_list;
   REQUIRE(pmesh.bdr_attributes.Max() > 0);

   Array<int> bdr_attr_is_ess(pmesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   if (disabled_boundary >= 0)
   {
      CAPTURE(disabled_boundary);
      bdr_attr_is_ess[pmesh.bdr_attributes.Find(disabled_boundary)] = 0;
   }

   pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   REQUIRE(ess_tdof_list.Size() > 0);

   ConstantCoefficient zero(0.0);
   sol.ProjectCoefficient(zero);
   Vector B, X;
   OperatorPtr A;
   const bool copy_interior = true; // interior(sol) --> interior(X)
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);

   // Solve the system
   CGSolver cg(MPI_COMM_WORLD);
   // HypreBoomerAMG preconditioner;
   cg.SetMaxIter(2000);
   cg.SetRelTol(1e-12);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   // cg.SetPreconditioner(preconditioner);
   cg.Mult(B, X);
   // Recover the solution
   a.RecoverFEMSolution(X, b, sol);

   // Check that X solves the system A X = B.
   A->AddMult(X, B, -1.0);
   auto residual_norm = B.Norml2();
   bool satisfy_system = residual_norm < 1e-10;
   CAPTURE(residual_norm);
   CHECK(satisfy_system);

   // Initialize the bdr_dof to be checked
   Vector tvec;
   sol.GetTrueDofs(tvec);
   bool satisfy_bc = true;
   for (auto dof : ess_tdof_list)
   {
      if (tvec[dof] != 0.0)
      {
         satisfy_bc = false;
         break;
      }
   }
   CHECK(satisfy_bc);
};

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh.
//            (note: permutations of the values in the diagonal are expected)
TEST_CASE("pNCMesh PA diagonal",  "[Parallel], [NCMesh]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   SECTION("Quad pmesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian2D(
                        ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      ParMesh nc_pmesh(MPI_COMM_WORLD, nc_mesh);

      int dim = 2;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         ParFiniteElementSpace pfes(&pmesh, &fec);
         ParFiniteElementSpace nc_pfes(&nc_pmesh, &fec);

         ParBilinearForm a(&pfes);
         ParBilinearForm nc_a(&nc_pfes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(pfes.GetTrueVSize());
         Vector nc_diag(nc_pfes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         double diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         double diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         double error = fabs(diag_gsum - nc_diag_gsum);
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

   SECTION("Hexa pmesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian3D(
                     ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian3D(
                        ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      ParMesh nc_pmesh(MPI_COMM_WORLD, nc_mesh);

      int dim = 3;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         ParFiniteElementSpace pfes(&pmesh, &fec);
         ParFiniteElementSpace nc_pfes(&nc_pmesh, &fec);

         ParBilinearForm a(&pfes);
         ParBilinearForm nc_a(&nc_pfes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(pfes.GetTrueVSize());
         Vector nc_diag(nc_pfes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         double diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         double diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         double error = fabs(diag_gsum - nc_diag_gsum);
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
} // test case


// Given a parallel and a serial mesh, perform an L2 projection and check the
// solutions match exactly.
void CheckL2Projection(ParMesh& pmesh, Mesh& smesh, int order,
                       std::function<double(Vector const&)> exact_soln)
{
   REQUIRE(pmesh.GetGlobalNE() == smesh.GetNE());
   REQUIRE(pmesh.Dimension() == smesh.Dimension());
   REQUIRE(pmesh.SpaceDimension() == smesh.SpaceDimension());

   // Make an H1 space, then a mass matrix operator and invert it.
   // If all non-conformal constraints have been conveyed correctly, the
   // resulting DOF should match exactly on the serial and the parallel
   // solution.

   H1_FECollection fec(order, smesh.Dimension());
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef(exact_soln);

   constexpr double linear_tol = 1e-16;

   // serial solve
   auto serror = [&]
   {
      FiniteElementSpace fes(&smesh, &fec);
      // solution vectors
      GridFunction x(&fes);
      x = 0.0;

      double snorm = x.ComputeL2Error(rhs_coef);

      LinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
      b.Assemble();

      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();

      SparseMatrix A;
      Vector B, X;

      Array<int> empty_tdof_list;
      a.FormLinearSystem(empty_tdof_list, x, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
      // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //    solve the system AX=B with PCG.
      GSSmoother M(A);
      PCG(A, M, B, X, -1, 500, linear_tol, 0.0);
#else
      // 9. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(B, X);
#endif

      a.RecoverFEMSolution(X, b, x);
      return x.ComputeL2Error(rhs_coef) / snorm;
   }();

   auto perror = [&]
   {
      // parallel solve
      ParFiniteElementSpace fes(&pmesh, &fec);
      ParLinearForm b(&fes);

      ParGridFunction x(&fes);
      x = 0.0;

      double pnorm = x.ComputeL2Error(rhs_coef);
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
      b.Assemble();

      ParBilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();

      HypreParMatrix A;
      Vector B, X;
      Array<int> empty_tdof_list;
      a.FormLinearSystem(empty_tdof_list, x, b, A, X, B);

      HypreBoomerAMG amg(A);
      HyprePCG pcg(A);
      amg.SetPrintLevel(-1);
      pcg.SetTol(linear_tol);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(-1);
      pcg.SetPreconditioner(amg);
      pcg.Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
      return x.ComputeL2Error(rhs_coef) / pnorm;
   }();

   constexpr double test_tol = 1e-9;
   CHECK(std::abs(serror - perror) < test_tol);
};

TEST_CASE("FaceEdgeConstraint",  "[Parallel], [NCMesh]")
{
   constexpr int refining_rank = 0;
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   REQUIRE(smesh.GetNE() == 1);
   {
      // Start the test with two tetrahedra attached by triangle.
      auto single_edge_refine = Array<Refinement>(1);
      single_edge_refine[0].index = 0;
      single_edge_refine[0].ref_type = Refinement::X;

      smesh.GeneralRefinement(single_edge_refine, 0); // conformal
   }

   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   REQUIRE(smesh.GetNE() == 2);
   smesh.EnsureNCMesh(true);
   smesh.Finalize();

   auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);
   partition[0] = 0;
   partition[1] = Mpi::WorldSize() > 1 ? 1 : 0;

   auto pmesh = ParMesh(MPI_COMM_WORLD, smesh, partition.get());

   // Construct the NC refined mesh in parallel and serial. Once constructed a
   // global L2 projected solution should match exactly on each.
   Array<int> refines, serial_refines(1);
   if (Mpi::WorldRank() == refining_rank)
   {
      refines.Append(0);
   }

   // Must be called on all ranks as it uses MPI calls internally.
   // All ranks will use the global element number dictated by rank 0 though.
   serial_refines[0] = pmesh.GetGlobalElementNum(0);
   MPI_Bcast(&serial_refines[0], 1, MPI_INT, refining_rank, MPI_COMM_WORLD);

   // Rank 0 refines the parallel mesh, all ranks refine the serial mesh
   smesh.GeneralRefinement(serial_refines, 1); // nonconformal
   pmesh.GeneralRefinement(refines, 1); // nonconformal

   REQUIRE(pmesh.GetGlobalNE() == 8 + 1);
   REQUIRE(smesh.GetNE() == 8 + 1);

   // Each pair of indices here represents sequential element indices to refine.
   // First the i element is refined, then in the resulting mesh the j element is
   // refined. These pairs were arrived at by looping over all possible i,j pairs and
   // checking for the addition of a face-edge constraint.
   std::vector<std::pair<int,int>> indices{{2,13}, {3,13}, {6,2}, {6,3}};

   // Rank 0 has all but one element in the parallel mesh. The remaining element
   // is owned by another processor if the number of ranks is greater than one.
   for (const auto &ij : indices)
   {
      int i = ij.first;
      int j = ij.second;
      if (Mpi::WorldRank() == refining_rank)
      {
         refines[0] = i;
      }
      // Inform all ranks of the serial mesh
      serial_refines[0] = pmesh.GetGlobalElementNum(i);
      MPI_Bcast(&serial_refines[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

      ParMesh tmp(pmesh);
      tmp.GeneralRefinement(refines);

      REQUIRE(tmp.GetGlobalNE() == 1 + 8 - 1 + 8); // 16 elements

      Mesh stmp(smesh);
      stmp.GeneralRefinement(serial_refines);
      REQUIRE(stmp.GetNE() == 1 + 8 - 1 + 8); // 16 elements

      if (Mpi::WorldRank() == refining_rank)
      {
         refines[0] = j;
      }
      // Inform all ranks of the serial mesh
      serial_refines[0] = tmp.GetGlobalElementNum(j);
      MPI_Bcast(&serial_refines[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

      ParMesh ttmp(tmp);
      ttmp.GeneralRefinement(refines);

      REQUIRE(ttmp.GetGlobalNE() == 1 + 8 - 1 + 8 - 1 + 8); // 23 elements

      Mesh sttmp(stmp);
      sttmp.GeneralRefinement(serial_refines);
      REQUIRE(sttmp.GetNE() == 1 + 8 - 1 + 8 - 1 + 8); // 23 elements

      // Loop over interior faces, fill and check face transform on the serial.
      for (int iface = 0; iface < sttmp.GetNumFaces(); ++iface)
      {
         const auto face_transform = sttmp.GetFaceElementTransformations(iface);
         CHECK(face_transform->CheckConsistency(0) < 1e-12);
      }

      for (int iface = 0; iface < ttmp.GetNumFacesWithGhost(); ++iface)
      {
         const auto face_transform = ttmp.GetFaceElementTransformations(iface);
         CHECK(face_transform->CheckConsistency(0) < 1e-12);
      }

      // Use P4 to ensure there's a few fully interior DOF.
      CheckL2Projection(ttmp, sttmp, 4, exact_soln);

      ttmp.ExchangeFaceNbrData();
      ttmp.Rebalance();

      CheckL2Projection(ttmp, sttmp, 4, exact_soln);
   }
} // test case

Mesh CylinderMesh(Geometry::Type el_type, bool quadratic, int variant = 0)
{
   double c[3];

   int nnodes = (el_type == Geometry::CUBE) ? 24 : 15;
   int nelems = 8; // Geometry::PRISM
   if (el_type == Geometry::CUBE)        { nelems = 10; }
   if (el_type == Geometry::TETRAHEDRON) { nelems = 24; }

   Mesh mesh(3, nnodes, nelems);

   for (int i=0; i<3; i++)
   {
      if (el_type != Geometry::CUBE)
      {
         c[0] = 0.0;  c[1] = 0.0;  c[2] = 2.74 * i;
         mesh.AddVertex(c);
      }

      for (int j=0; j<4; j++)
      {
         if (el_type == Geometry::CUBE)
         {
            c[0] = 1.14 * ((j + 1) % 2) * (1 - j);
            c[1] = 1.14 * (j % 2) * (2 - j);
            c[2] = 2.74 * i;
            mesh.AddVertex(c);
         }

         c[0] = 2.74 * ((j + 1) % 2) * (1 - j);
         c[1] = 2.74 * (j % 2) * (2 - j);
         c[2] = 2.74 * i;
         mesh.AddVertex(c);
      }
   }

   for (int i=0; i<2; i++)
   {
      if (el_type == Geometry::CUBE)
      {
         mesh.AddHex(8*i, 8*i+2, 8*i+4, 8*i+6,
                     8*(i+1), 8*(i+1)+2, 8*(i+1)+4, 8*(i+1)+6);
      }

      for (int j=0; j<4; j++)
      {
         if (el_type == Geometry::PRISM)
         {
            switch (variant)
            {
               case 0:
                  mesh.AddWedge(5*i, 5*i+j+1, 5*i+(j+1)%4+1,
                                5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
                  break;
               case 1:
                  mesh.AddWedge(5*i, 5*i+j+1, 5*i+(j+1)%4+1,
                                5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
                  break;
               case 2:
                  mesh.AddWedge(5*i+(j+1)%4+1, 5*i, 5*i+j+1,
                                5*(i+1)+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1);
                  break;
            }
         }
         else if (el_type == Geometry::CUBE)
         {
            mesh.AddHex(8*i+2*j, 8*i+2*j+1, 8*i+(2*j+3)%8, 8*i+(2*j+2)%8,
                        8*(i+1)+2*j, 8*(i+1)+2*j+1, 8*(i+1)+(2*j+3)%8,
                        8*(i+1)+(2*j+2)%8);
         }
         else if (el_type == Geometry::TETRAHEDRON)
         {
            mesh.AddTet(5*i, 5*i+j+1, 5*i+(j+1)%4+1, 5*(i+1));
            mesh.AddTet(5*i+j+1, 5*i+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1);
            mesh.AddTet(5*i+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
         }
      }
   }

   mesh.FinalizeTopology();

   if (quadratic)
   {
      mesh.SetCurvature(2);

      if (el_type == Geometry::CUBE)
      {
         auto quad_cyl_hex = [](const Vector& x, Vector& d)
         {
            d.SetSize(3);
            d = x;
            const double Rmax = 2.74;
            const double Rmin = 1.14;
            double ax = std::abs(x[0]);
            if (ax <= 1e-6) { return; }
            double ay = std::abs(x[1]);
            if (ay <= 1e-6) { return; }
            double r = ax + ay;
            if (r <= Rmin + 1e-6) { return; }

            double sx = std::copysign(1.0, x[0]);
            double sy = std::copysign(1.0, x[1]);

            double R = (Rmax - Rmin) * Rmax / (r - Rmin);
            double r2 = r * r;
            double R2 = R * R;

            double acosarg = 0.5 * (r + std::sqrt(2.0 * R2 - r2)) / R;
            double tR = std::acos(std::min(acosarg, 1.0));
            double tQ = (1.0 + sx * sy * (ay - ax) / r);
            double tP = 0.25 * M_PI * (3.0 - (2.0 + sx) * sy);

            double t = tR + (0.25 * M_PI - tR) * tQ + tP;

            double s0 = std::sqrt(2.0 * R2 - r2);
            double s1 = 0.25 * std::pow(r + s0, 2);
            double s = std::sqrt(R2 - s1);

            d[0] = R * std::cos(t) - sx * s;
            d[1] = R * std::sin(t) - sy * s;

            return;
         };

         mesh.Transform(quad_cyl_hex);
      }
      else
      {
         auto quad_cyl = [](const Vector& x, Vector& d)
         {
            d.SetSize(3);
            d = x;
            double ax = std::abs(x[0]);
            double ay = std::abs(x[1]);
            double r = ax + ay;
            if (r < 1e-6) { return; }

            double sx = std::copysign(1.0, x[0]);
            double sy = std::copysign(1.0, x[1]);

            double t = ((2.0 - (1.0 + sx) * sy) * ax +
                        (2.0 - sy) * ay) * 0.5 * M_PI / r;
            d[0] = r * std::cos(t);
            d[1] = r * std::sin(t);

            return;
         };

         mesh.Transform(quad_cyl);
      }
   }

   mesh.Finalize(true);

   return mesh;
}

TEST_CASE("P2Q1PureTetHexPri",  "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   auto el_type = GENERATE(Geometry::TETRAHEDRON,
                           Geometry::CUBE,
                           Geometry::PRISM);
   int variant = GENERATE(0,1,2);

   if (variant > 0 && el_type != Geometry::PRISM)
   {
      return;
   }

   CAPTURE(el_type, variant);

   auto smesh = CylinderMesh(el_type, false, variant);

   for (auto ref : {0,1,2})
   {
      if (ref == 1) { smesh.UniformRefinement(); }

      smesh.EnsureNCMesh(true);

      if (ref == 2) { smesh.UniformRefinement(); }

      smesh.Finalize();

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

      // P2 ensures there are triangles without dofs
      CheckL2Projection(pmesh, smesh, 2, exact_soln);
   }
} // test case

TEST_CASE("PNQ2PureTetHexPri",  "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   auto el_type = GENERATE(Geometry::TETRAHEDRON,
                           Geometry::CUBE,
                           Geometry::PRISM);
   int variant = GENERATE(0,1,2);

   if (variant > 0 && el_type != Geometry::PRISM)
   {
      return;
   }

   CAPTURE(el_type, variant);

   auto smesh = CylinderMesh(el_type, true);

   for (auto ref : {0,1,2})
   {
      if (ref == 1) { smesh.UniformRefinement(); }

      smesh.EnsureNCMesh(true);

      if (ref == 2) { smesh.UniformRefinement(); }

      smesh.Finalize();

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

      for (int p = 1; p < 3; ++p)
      {
         CheckL2Projection(pmesh, smesh, p, exact_soln);
      }
   }
} // test case

/**
 * @brief Test GetVectorValue on face neighbor elements for nonconformal meshes
 *
 * @param smesh The serial mesh to start from
 * @param nc_level Depth of refinement on processor boundaries
 * @param skip Refine every "skip" processor boundary element
 * @param use_ND Whether to use Nedelec elements (which are sensitive to orientation)
 */
void TestVectorValueInVolume(Mesh &smesh, int nc_level, int skip, bool use_ND)
{
   auto vector_exact_soln = [](const Vector& x, Vector& v)
   {
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      v = (d -= x);
   };

   smesh.Finalize();
   smesh.EnsureNCMesh(true); // uncomment this to trigger the failure

   auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

   // Apply refinement on face neighbors to achieve a given nc level mismatch.
   for (int i = 0; i < nc_level; ++i)
   {
      // To refine the face neighbors, need to know where they are.
      pmesh.ExchangeFaceNbrData();
      Array<int> elem_to_refine;
      // Refine only on odd ranks.
      if ((Mpi::WorldRank() + 1) % 2 == 0)
      {
         // Refine a subset of all shared faces. Using a subset helps to
         // mix in conformal faces with nonconformal faces.
         for (int n = 0; n < pmesh.GetNSharedFaces(); ++n)
         {
            if (n % skip != 0) { continue; }
            const int local_face = pmesh.GetSharedFace(n);
            const auto &face_info = pmesh.GetFaceInformation(local_face);
            REQUIRE(face_info.IsShared());
            REQUIRE(face_info.element[1].location == Mesh::ElementLocation::FaceNbr);
            elem_to_refine.Append(face_info.element[0].index);
         }
      }
      pmesh.GeneralRefinement(elem_to_refine);
   }

   // Do not rebalance again! The test is also checking for nc refinements
   // along the processor boundary.

   // Create a grid function of the mesh coordinates
   pmesh.ExchangeFaceNbrData();
   pmesh.EnsureNodes();
   REQUIRE(pmesh.OwnsNodes());
   GridFunction * const coords = pmesh.GetNodes();
   dynamic_cast<ParGridFunction *>(pmesh.GetNodes())->ExchangeFaceNbrData();

   // Project the linear function onto the mesh. Quadratic ND tetrahedral
   // elements are the first to require face orientations.
   const int order = 2, dim = 3;
   std::unique_ptr<FiniteElementCollection> fec;
   if (use_ND)
   {
      fec = std::unique_ptr<ND_FECollection>(new ND_FECollection(order, dim));
   }
   else
   {
      fec = std::unique_ptr<RT_FECollection>(new RT_FECollection(order, dim));
   }
   ParFiniteElementSpace pnd_fes(&pmesh, fec.get());

   ParGridFunction psol(&pnd_fes);

   VectorFunctionCoefficient func(3, vector_exact_soln);
   psol.ProjectCoefficient(func);
   psol.ExchangeFaceNbrData();

   mfem::Vector value(3), exact(3), position(3);
   const IntegrationRule &ir = mfem::IntRules.Get(Geometry::Type::TETRAHEDRON,
                                                  order + 1);

   // Check that non-ghost elements match up on the serial and parallel spaces.
   for (int n = 0; n < pmesh.GetNE(); ++n)
   {
      constexpr double tol = 1e-12;
      for (const auto &ip : ir)
      {
         coords->GetVectorValue(n, ip, position);
         psol.GetVectorValue(n, ip, value);

         vector_exact_soln(position, exact);

         REQUIRE(value.Size() == exact.Size());
         CHECK((value -= exact).Normlinf() < tol);
      }
   }

   // Loop over face neighbor elements and check the vector values match in the
   // face neighbor elements.
   for (int n = 0; n < pmesh.GetNSharedFaces(); ++n)
   {
      const int local_face = pmesh.GetSharedFace(n);
      const auto &face_info = pmesh.GetFaceInformation(local_face);
      REQUIRE(face_info.IsShared());
      REQUIRE(face_info.element[1].location == Mesh::ElementLocation::FaceNbr);

      auto &T = *pmesh.GetFaceNbrElementTransformation(face_info.element[1].index);

      constexpr double tol = 1e-12;
      for (const auto &ip : ir)
      {
         T.SetIntPoint(&ip);
         coords->GetVectorValue(T, ip, position);
         psol.GetVectorValue(T, ip, value);

         vector_exact_soln(position, exact);

         REQUIRE(value.Size() == exact.Size());
         CHECK((value -= exact).Normlinf() < tol);
      }
   }
}

TEST_CASE("GetVectorValueInFaceNeighborElement", "[Parallel], [NCMesh]")
{
   // The aim of this test is to verify the correct behaviour of the
   // GetVectorValue method when called on face neighbor elements in a non
   // conforming mesh.
   auto smesh = Mesh("../../data/beam-tet.mesh");

   for (int nc_level : {0,1,2,3})
   {
      for (int skip : {1,2})
      {
         for (bool use_ND : {false, true})
         {
            TestVectorValueInVolume(smesh, nc_level, skip, use_ND);
         }
      }
   }
}

TEST_CASE("InteriorBoundaryElementConsistency", "[Parallel], [NCMesh]")
{
   auto p = GENERATE(2);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-cube.mesh");
   smesh.EnsureNCMesh();

   REQUIRE(smesh.GetNBE() == 6);

   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(0, 1);
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 5);

   // Throw away the NCMesh, will restart NC later.
   delete smesh.ncmesh;
   smesh.ncmesh = nullptr;

   // Introduce an internal boundary element
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(7);
         smesh.AddBdrElement(new_elem);
         break;
      }
   }
   smesh.FinalizeTopology(); // Finalize to build relevant tables

   // Exactly one boundary element must be added
   REQUIRE(smesh.GetNBE() == 2 * 5 + 1);

   auto check_parmesh = [](Mesh &smesh,
                           const std::unique_ptr<int[]> &partition = nullptr)
   {
      auto pmesh = std::unique_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, smesh,
                                                        partition.get()));

      int nbe = pmesh->GetNBE();
      MPI_Allreduce(MPI_IN_PLACE, &nbe, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      CHECK(nbe == smesh.GetNBE());
      return pmesh;
   };

   auto pmesh = check_parmesh(smesh);

   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Conforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   auto check_face_internal = [](ParMesh& pmesh, int f,
                                 const std::map<int, int> &local_to_shared)
   {
      int e1, e2;
      pmesh.GetFaceElements(f, &e1, &e2);
      int inf1, inf2, ncface;
      pmesh.GetFaceInfos(f, &inf1, &inf2, &ncface);

      if (e2 < 0 && inf2 >=0)
      {
         // Shared face on processor boundary -> Need to discover the neighbor
         // attributes
         auto FET = pmesh.GetSharedFaceTransformations(local_to_shared.at(f));
         if (FET->Elem1->Attribute != FET->Elem2->Attribute
             && (Mpi::WorldRank() < pmesh.GetFaceNbrRank(FET->Elem2No - pmesh.GetNE()) ||
                 ncface >= 0))
         {
            // shared face on domain attribute boundary, which we are the owner
            // of, or is a slave face
            return true;
         }
      }

      if (e2 >= 0 && pmesh.GetAttribute(e1) != pmesh.GetAttribute(e2))
      {
         // local face on domain attribute boundary
         return true;
      }
      return false;
   };

   int num_internal = 0;
   // Count the number of internal faces via the faces
   for (int f = 0; f < pmesh->GetNumFaces(); ++f)
   {
      if (check_face_internal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   // Count the number of internal faces via the boundary elements
   num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (check_face_internal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   smesh.EnsureNCMesh(true);
   smesh.FinalizeTopology();

   check_parmesh(smesh);
   // CheckPoisson(*pmesh, p, 7);
   // CheckPoisson(*pmesh, p);

   for (int refined_elem : {0, 1})
   {
      // Now NC refine one of the attached elements, this should result in 4
      // internal boundary elements.
      Array<int> el_to_refine;
      el_to_refine.Append(refined_elem);

      Mesh modified_smesh(smesh);
      modified_smesh.GeneralRefinement(el_to_refine);

      // There should now be four internal boundary elements, where there was one
      // before.
      CHECK(modified_smesh.GetNBE() == 5 /* external boundaries of unrefined  */
            + 4 /* internal boundaries */
            + (5 * 4) /* external boundaries of refined */);

      auto partition = std::unique_ptr<int[]>(new int[modified_smesh.GetNE()]);
      for (int i = 0; i < modified_smesh.GetNE(); ++i)
      {
         partition[i] = 0;
      }
      partition[refined_elem + 1 % 2] = Mpi::WorldSize() > 1 ? 1 : 0;

      std::cout << "Finalizing the ParMesh\n";

      pmesh->Finalize();
      pmesh->FinalizeTopology();
      pmesh->ExchangeFaceNbrData();
      pmesh = check_parmesh(modified_smesh, partition);

      // repopulate the local to shared map.
      local_to_shared.clear();
      for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
      {
         local_to_shared[pmesh->GetSharedFace(i)] = i;
      }

      // Count the number of internal faces via the boundary elements
      num_internal = 0;
      for (int n = 0; n < pmesh->GetNBE(); ++n)
      {
         int f, o;
         pmesh->GetBdrElementFace(n, &f, &o);
         if (check_face_internal(*pmesh, f, local_to_shared))
         {
            ++num_internal;
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK(num_internal == 4);

      // Define a space and apply an essential boundary condition on the
      // internal boundary.
      CAPTURE(refined_elem);

      CheckPoisson(*pmesh, p, 7);
      CheckPoisson(*pmesh, p);
   }
}

#endif // MFEM_USE_MPI

TEST_CASE("ReferenceCubeInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-cube.mesh");
   smesh.EnsureNCMesh();
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 5);

   delete smesh.ncmesh;
   smesh.ncmesh = nullptr;

   // Introduce an internal boundary element
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(7);
         smesh.AddBdrElement(new_elem);
         break;
      }
   }

   smesh.FinalizeTopology();

   // Exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 5 + 1);

   smesh.EnsureNCMesh();
   CHECK(smesh.GetNBE() == 2 * 5 + 1);

   int without_internal = CheckPoisson(smesh, p, 7); // Exclude the internal boundary
   int with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 1); break;
      case 3:
         CHECK(with_internal == without_internal + 4); break;
   }

   auto ref_type = char(GENERATE(Refinement::Y, Refinement::Z, Refinement::YZ,
                                 Refinement::XYZ));

   for (auto ref : {0,1})
   {
      refs[0].index = ref;

      auto ssmesh = Mesh(smesh);

      CAPTURE(ref_type);

      // Now NC refine one of the attached elements, this should result in 2
      // internal boundary elements.
      refs[0].ref_type = ref_type;

      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was one
      // before.
      if (ref_type == 2 /* Y */ || ref_type == 4 /* Z */)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 2 /* internal boundaries */
               + (2 * 4) /* external boundaries of refined elements */);
      }
      else if (ref_type == 6)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 4 /* internal boundaries */
               + (4 * 3) /* external boundaries of refined elements */);
      }
      else if (ref_type == 7)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 4 /* internal boundaries */
               + (4 * 3 + 4 * 2) /* external boundaries of refined elements */);
      }
      else
      {
         MFEM_ABORT("!");
      }

      // Count the number of internal boundary elements
      int num_internal = 0;
      for (int n = 0; n < ssmesh.GetNBE(); ++n)
      {
         int f, o;
         ssmesh.GetBdrElementFace(n, &f, &o);
         int e1, e2;
         ssmesh.GetFaceElements(f, &e1, &e2);
         if (e1 >= 0 && e2 >= 0 && ssmesh.GetAttribute(e1) != ssmesh.GetAttribute(e2))
         {
            ++num_internal;
         }
      }
      CHECK(num_internal == (ref_type <= 4 ? 2 : 4));

      ssmesh.FinalizeTopology();

      without_internal = CheckPoisson(ssmesh, p, 7); // Exclude the internal boundary
      with_internal = CheckPoisson(ssmesh, p); // Include the internal boundary

      switch (p)
      {
         case 1:
            CHECK(with_internal == without_internal + 1*1); break;
         case 2:
            CHECK(with_internal == without_internal + 3*3); break;
         case 3:
            CHECK(with_internal == without_internal + 5*5); break;
      }
   }
}

TEST_CASE("ReferenceTetInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);
   {
      auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

      CHECK(smesh.GetNBE() == 4);

      smesh.UniformRefinement();

      CHECK(smesh.GetNBE() == 4 * 4);

      CHECK(smesh.Conforming());
   }

   {
      auto smesh = Mesh("../../data/ref-tetrahedron.mesh");
      smesh.EnsureNCMesh(true);

      CHECK(smesh.GetNBE() == 4);

      smesh.UniformRefinement();

      CHECK(smesh.GetNBE() == 4 * 4);

      CHECK(smesh.Nonconforming());
   }

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNE() == 2);
   REQUIRE(smesh.GetNBE() == 2 * 3);

   // Introduce an internal boundary element
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(4);
         smesh.AddBdrElement(new_elem);
         break;
      }
   }

   // Exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 + 1);

   smesh.EnsureNCMesh(true);

   // Still exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 + 1);

   smesh.FinalizeTopology();
   smesh.Finalize();

   CheckPoisson(smesh, p, 4); // Exclude the internal boundary
   CheckPoisson(smesh, p); // Include the internal boundary

   // Now NC refine one of the attached elements, this should result in 2
   // internal boundary elements.
   for (int ref : {0, 1})
   {
      refs[0].index = ref;
      refs[0].ref_type = Refinement::XYZ;
      auto ssmesh = Mesh(smesh);
      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was one
      // before.
      CHECK(ssmesh.GetNBE() == 3 /* external boundaries of unrefined element  */
            + 4 /* internal boundaries */
            + (3 * 4) /* external boundaries of refined element */);

      // Count the number of internal boundary elements
      int num_internal = 0;
      for (int n = 0; n < ssmesh.GetNBE(); ++n)
      {
         int f, o;
         ssmesh.GetBdrElementFace(n, &f, &o);
         int e1, e2;
         ssmesh.GetFaceElements(f, &e1, &e2);
         if (e1 >= 0 && e2 >= 0 && ssmesh.GetAttribute(e1) != ssmesh.GetAttribute(e2))
         {
            ++num_internal;
         }
      }
      CHECK(num_internal == 4);

      CheckPoisson(ssmesh, p, 4); // Exclude the internal boundary
      CheckPoisson(ssmesh, p); // Include the internal boundary
   }
}

TEST_CASE("PoissonOnReferenceCubeNC", "[NCMesh]")
{
   auto smesh = Mesh("../../data/ref-cube.mesh");
   smesh.EnsureNCMesh();
   Array<Refinement> refs(1);
   refs[0].index = 0;
   refs[0].ref_type = Refinement::X;
   smesh.GeneralRefinement(refs);

   // Now have two elements.
   smesh.FinalizeTopology();
   smesh.Finalize();

   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);

   // Check that Poisson can be solved on the domain
   CheckPoisson(smesh, p);

   auto ref_type = char(GENERATE(Refinement::X, Refinement::Y, Refinement::Z,
                                 Refinement::XY, Refinement::XZ, Refinement::YZ,
                                 Refinement::XYZ));
   CAPTURE(ref_type);
   for (auto refined_elem : {0}) // The left or the right element
   {
      refs[0].index = refined_elem;
      auto ssmesh = Mesh(smesh);

      // Now NC refine one of the attached elements
      refs[0].ref_type = ref_type;

      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

TEST_CASE("PoissonOnReferenceTetNC", "[NCMesh]")
{
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);

   CheckPoisson(smesh, p);

   Array<Refinement> refs(1);
   refs[0].index = 0;
   refs[0].ref_type = Refinement::X;

   smesh.GeneralRefinement(refs);

   // Now have two elements.
   smesh.FinalizeTopology();
   smesh.Finalize();

   // Check that Poisson can be solved on the pair of tets
   CheckPoisson(smesh, p);

   auto nc = GENERATE(false, true);
   CAPTURE(nc);

   smesh.EnsureNCMesh(GENERATE(false, true));

   for (auto refined_elem : {0, 1})
   {
      auto ssmesh = Mesh(smesh);

      refs[0].index = refined_elem;
      refs[0].ref_type = Refinement::XYZ;

      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

} // namespace mfem
