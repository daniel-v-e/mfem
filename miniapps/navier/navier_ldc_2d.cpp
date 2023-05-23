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
//
// Navier 2D lid-driven cavity flow
//    based on MMS example
//


#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 5;
   double kinvis = 1.0;
   double t_final = 50 * 0.25e-2;
   double dt = 0.25e-2;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
} ctx;

void vel(const Vector &x, double t, Vector &u)
{
   // double xi = x(0);
   double yi = x(1);

   if (abs(yi - 1.0) < 1e-8) { u(0) = 1.0; u(1) = 0.0; } else { u = 0.0; }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pa",
                  "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   Mesh *mesh = new Mesh("../../data/inline-quad.mesh");
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= 2.0;
   *nodes -= 1.0;

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver naviersolver(pmesh, ctx.order, ctx.kinvis);
   naviersolver.EnablePA(ctx.pa);
   naviersolver.EnableNI(ctx.ni);

   // Set the initial condition.
   ParGridFunction *u_ic = naviersolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   u_ic->ProjectCoefficient(u_excoeff);

   // FunctionCoefficient p_excoeff(p);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   naviersolver.AddVelDirichletBC(vel, attr);

   // Array<int> domain_attr(pmesh->attributes.Max());
   // domain_attr = 1;
   // naviersolver.AddAccelTerm(accel, domain_attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   naviersolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   u_gf = naviersolver.GetCurrentVelocity();
   p_gf = naviersolver.GetCurrentPressure();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      naviersolver.Step(t, dt, step);

      // Compare against exact solution of velocity and pressure.
      // u_excoeff.SetTime(t);
      // p_excoeff.SetTime(t);
      // err_u = u_gf->ComputeL2Error(u_excoeff);
      // err_p = p_gf->ComputeL2Error(p_excoeff);

      if (Mpi::Root())
      {
         printf("%11s %11s %11s %11s\n", "Time", "dt", "err_u", "err_p");
         printf("%.5E %.5E %.5E %.5E err\n", t, dt, err_u, err_p);
         fflush(stdout);
      }
   }

   ParaViewDataCollection pvdc("ldc_2d_output", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   // pvdc.RegisterField("vorticity", &w_gf);
   // pvdc.RegisterField("qcriterion", &q_gf);
   pvdc.Save();

   if (ctx.visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " "
               << Mpi::WorldRank() << "\n";
      sol_sock << "solution\n" << *pmesh << *u_ic << std::flush;
   }

   naviersolver.PrintTimingData();

   // Test if the result for the test run is as expected.
   if (ctx.checkres)
   {
      double tol = 1e-3;
      if (err_u > tol || err_p > tol)
      {
         if (Mpi::Root())
         {
            mfem::out << "Result has a larger error than expected."
                      << std::endl;
         }
         return -1;
      }
   }

   delete pmesh;

   return 0;
}