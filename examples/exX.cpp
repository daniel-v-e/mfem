//                                MFEM Example X
//
// Compile with: make ex9
//
// Sample runs:
//    exX
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. The saving of time-dependent data files for external
//               visualization with VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) is also illustrated.

#include "mfem.hpp"
#include "proximalGalerkin.hpp"

// Solution variables
class Vars { public: enum {u, f_rho, psi, f_lam, numVars}; };

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 0;
   const char *mesh_file = "../data/rect_with_top_fixed.mesh";
   int ref_levels = 2;
   int order = 3;
   const char *device_config = "cpu";
   bool visualization = true;
   double alpha0 = 1.0;
   double epsilon = 1e-03;
   double rho0 = 1e-6;
   int simp_exp = 3;


   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();
   const int max_attributes = mesh.bdr_attributes.Max();

   double volume = 0.0;
   for (int i=0; i<mesh.GetNE(); i++) { volume += mesh.GetElementVolume(i); }

   for (int i=0; i<ref_levels; i++) { mesh.UniformRefinement(); }

   // Essential boundary for each variable (numVar x numAttr)
   Array2D<int> ess_bdr(Vars::numVars, max_attributes);
   ess_bdr = 0;
   ess_bdr(Vars::u, 0) = true;

   // Source and fixed temperature
   ConstantCoefficient heat_source(1.0);
   ConstantCoefficient u_bdr(0.0);
   const double volume_fraction = 0.7;
   const double target_volume = volume * volume_fraction;

   // Finite Element Spaces
   FiniteElementSpace fes_H1_Qk2(&mesh, new H1_FECollection(order + 2, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_H1_Qk1(&mesh, new H1_FECollection(order + 1, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_H1_Qk0(&mesh, new H1_FECollection(order + 0, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_L2_Qk2(&mesh, new L2_FECollection(order + 2, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_L2_Qk1(&mesh, new L2_FECollection(order + 1, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_L2_Qk0(&mesh, new L2_FECollection(order + 0, dim,
                                                            mfem::BasisType::GaussLobatto));

   Array<FiniteElementSpace*> fes(Vars::numVars);
   fes[Vars::u] = &fes_H1_Qk1;
   fes[Vars::f_rho] = &fes_H1_Qk1;
   fes[Vars::psi] = &fes_L2_Qk0;
   fes[Vars::f_lam] = fes[Vars::f_rho];

   Array<int> offsets = getOffsets(fes);
   BlockVector sol(offsets), delta_sol(offsets);
   sol = 0.0;
   delta_sol = 0.0;

   GridFunction u(fes[Vars::u], sol.GetBlock(Vars::u));
   GridFunction psi(fes[Vars::psi], sol.GetBlock(Vars::psi));
   GridFunction f_rho(fes[Vars::f_rho], sol.GetBlock(Vars::f_rho));
   GridFunction f_lam(fes[Vars::f_lam], sol.GetBlock(Vars::f_lam));

   GridFunction psi_k(fes[Vars::psi]);

   // Project solution
   Array<int> ess_bdr_u;
   ess_bdr_u.MakeRef(ess_bdr[Vars::u], max_attributes);
   u.ProjectBdrCoefficient(u_bdr, ess_bdr_u);
   psi = logit(volume_fraction);
   psi_k = logit(volume_fraction);
   f_rho = volume_fraction;

   return 0;
}

