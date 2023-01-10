#ifndef MHD_INITIAL_CONDITIONS_HPP
#define MHD_INITIAL_CONDITIONS_HPP

#include "mfem.hpp"

using namespace std;
using namespace mfem;

extern double beta;     //a global value of magnitude for the perturbation
extern double Lx;       //size of x domain
extern double lambda;
extern double resiG;
extern double ep;
extern double x_factor; //perturbation factor in x
extern double L0;       //initial current sheet width

//initial condition
double InitialPhi(const Vector &x);
double InitialW(const Vector &x);
double InitialJ(const Vector &x);
double InitialPsi(const Vector &x);
double BackPsi(const Vector &x);

//exact solution
double exactPhi1(const Vector &x, double t);
double exactPsi1(const Vector &x, double t);
double exactW1(const Vector &x, double t);

//exact solution for Reyleigh
double exactPhiRe(const Vector &x, double t);
double exactPsiRe(const Vector &x, double t);
double exactWRe(const Vector &x, double t);

double InitialJ2(const Vector &x);
double InitialPsi2(const Vector &x);
double BackPsi2(const Vector &x);

double E0rhs1(const Vector &x, double t);
double E0rhs(const Vector &x);
double E0rhs3(const Vector &x);
double E0rhs5(const Vector &x);

double InitialJ3(const Vector &x);
double InitialPsi3(const Vector &x);
double BackPsi3(const Vector &x);

double InitialPsi32(const Vector &x);
double BackPsi32(const Vector &x);

double InitialJ4(const Vector &x);
double InitialPsi4(const Vector &x);

double InitialJ6(const Vector &x);
double InitialPsi6(const Vector &x);

double InitialPsi7(const Vector &x);
double BackPsi7(const Vector &x);
double InitialJ7(const Vector &x);
double E0rhs7(const Vector &x);

double InitialPsi8(const Vector &x);
double BackPsi8(const Vector &x);
double InitialJ8(const Vector &x);

double InitialPsi9(const Vector &x);
double InitialJ9(const Vector &x);
double InitialPhi9(const Vector &x);
double InitialW9(const Vector &x);

double resiVari(const Vector &x);

#endif
