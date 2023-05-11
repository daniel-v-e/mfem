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

#include "spacing.hpp"

namespace mfem
{

SpacingFunction* GetSpacingFunction(const SPACING_TYPE spacingType,
                                    Array<int> const& ipar,
                                    Array<double> const& dpar)
{
   switch (spacingType)
   {
      case SPACING_TYPE::UNIFORM:
         MFEM_VERIFY(ipar.Size() == 1 &&
                     dpar.Size() == 0, "Invalid spacing function parameters");
         return new UniformSpacingFunction(ipar[0]);
      case SPACING_TYPE::LINEAR:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 1, "Invalid spacing function parameters");
         return new LinearSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                          (bool) ipar[2]);
      case SPACING_TYPE::GEOMETRIC:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 1, "Invalid spacing function parameters");
         return new GeometricSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                             (bool) ipar[2]);
      case SPACING_TYPE::BELL:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 2, "Invalid spacing function parameters");
         return new BellSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                        dpar[1], (bool) ipar[2]);
      case SPACING_TYPE::GAUSSIAN:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 2, "Invalid spacing function parameters");
         return new GaussianSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                            dpar[1], (bool) ipar[2]);
      case SPACING_TYPE::LOGARITHMIC:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 1, "Invalid spacing function parameters");
         return new LogarithmicSpacingFunction(ipar[0], (bool) ipar[1],
                                               (bool) ipar[2], dpar[0]);
      default:
         MFEM_ABORT("Unknown spacing type \"" << spacingType << "\"");
         break;
   }

   MFEM_ABORT("Unknown spacing type");
   return nullptr;
}

void GeometricSpacingFunction::CalculateSpacing()
{
   MFEM_VERIFY(n > 1, "GeometricSpacingFunction requires more than 1 interval");

   // Find the root of g(r) = s * (r^n - 1) - r + 1 by Newton's method.

   constexpr double convTol = 1.0e-8;
   constexpr int maxIter = 20;

   const double s_unif = 1.0 / ((double) n);

   r = s < s_unif ? 1.5 : 0.5;  // Initial guess

   bool converged = false;
   for (int iter=0; iter<maxIter; ++iter)
   {
      const double g = (s * (std::pow(r,n) - 1.0)) - r + 1.0;
      const double dg = (n * s * std::pow(r,n-1)) - 1.0;
      r -= g / dg;

      if (std::abs(g / dg) < convTol)
      {
         converged = true;
         break;
      }
   }

   MFEM_VERIFY(converged, "Convergence failure in GeometricSpacingFunction");
}

void BellSpacingFunction::CalculateSpacing()
{
   MFEM_VERIFY(n > 2, "Bell spacing requires at least three intervals");
   MFEM_VERIFY(s0 + s1 < 1.0, "Sum of first and last Bell spacings must be"
               << " less than 1");

   s.SetSize(n);
   s[0] = s0;
   s[n-1] = s1;

   // If there are only 3 intervals, the calculation is linear and trivial.
   if (n == 3)
   {
      s[1] = 1.0 - s0 - s1;
      return;
   }

   // For more than 3 intervals, solve a system iteratively.
   double urk = 1.0;

   // Initialize unknown entries of s.
   double initialGuess = (1.0 - s0 - s1) / ((double) (n - 2));
   for (int i=1; i<n-1; ++i)
   {
      s[i] = initialGuess;
   }

   Vector wk(7);
   wk = 0.0;

   Vector s_new(n);

   Vector a(n+2);
   Vector b(n+2);
   Vector alpha(n+2);
   Vector beta(n+2);
   Vector gamma(n+2);

   a = 0.5;
   a[0] = 0.0;
   a[1] = 0.0;

   b = a;

   alpha = 0.0;
   beta = 0.0;
   gamma = 0.0;

   gamma[1] = s0;

   constexpr int maxIter = 100;
   constexpr double convTol = 1.0e-10;
   bool converged = false;
   for (int iter=0; iter<maxIter; ++iter)
   {
      int j;
      for (j = 1; j <= n - 3; j++)
      {
         wk[0] = (s[j] + s[j+1]) * (s[j] + s[j+1]);
         wk[1] = s[j-1];
         wk[2] = (s[j-1] + s[j]) * (s[j-1] + s[j]) * (s[j-1] + s[j]);
         wk[3] = s[j + 2];
         wk[4] = (s[j+2] + s[j+1]) * (s[j+2] + s[j+1]) * (s[j+2] + s[j+1]);
         wk[5] = wk[0] * wk[1] / wk[2];
         wk[6] = wk[0] * wk[3] / wk[4];
         a[j+1]  = a[j+1] + urk*(wk[5] - a[j+1]);
         b[j+1]  = b[j+1] + urk*(wk[6] - b[j+1]);
      }

      for (j = 2; j <= n - 2; j++)
      {
         wk[0] = a[j]*(1.0 - 2.0*alpha[j - 1] + alpha[j - 1]*alpha[j - 2]
                       + beta[j - 2]) + b[j] + 2.0 - alpha[j - 1];
         wk[1] = 1.0 / wk[0];
         alpha[j] = wk[1]*(a[j]*beta[j - 1]*(2.0 - alpha[j - 2]) +
                           2.0*b[j] + beta[j - 1] + 1.0);
         beta[j]  = -b[j]*wk[1];
         gamma[j] = wk[1]*(a[j]*(2.0*gamma[j - 1] - gamma[j - 2] -
                                 alpha[j - 2]*gamma[j - 1]) + gamma[j - 1]);
      }

      s_new[0] = s[0];
      for (j=1; j<n; ++j)
      {
         s_new[j] = s_new[j-1] + s[j];
      }

      for (j = n - 3; j >= 1; j--)
      {
         s_new[j] = alpha[j+1]*s_new[j + 1] +
                    beta[j+1]*s_new[j + 2] + gamma[j+1];
      }

      // Convert back from points to spacings
      for (j=n-1; j>0; --j)
      {
         s_new[j] = s_new[j] - s_new[j-1];
      }

      wk[5] = wk[6] = 0.0;
      for (j = n - 2; j >= 2; j--)
      {
         wk[5] = wk[5] + s_new[j]*s_new[j];
         wk[6] = wk[6] + pow(s_new[j] - s[j], 2);
      }

      s = s_new;

      if (sqrt(wk[6] / wk[5]) < convTol)
      {
         converged = true;
         break;
      }
   }

   MFEM_VERIFY(converged, "Convergence failure in BellSpacingFunction");
}

void GaussianSpacingFunction::CalculateSpacing()
{
   MFEM_VERIFY(n > 2, "Gaussian spacing requires at least three intervals");
   MFEM_VERIFY(s0 + s1 < 1.0, "Sum of first and last Gaussian spacings must"
               << " be less than 1");

   s.SetSize(n);
   s[0] = s0;
   s[n-1] = s1;

   // If there are only 3 intervals, the calculation is linear and trivial.
   if (n == 3)
   {
      s[1] = 1.0 - s0 - s1;
      return;
   }

   // For more than 3 intervals, solve a system iteratively.

   const double lnz01 = log(s0 / s1);

   const double h = 1.0 / ((double) n-1);

   // Determine concavity by first determining linear spacing and comparing
   // the total spacing to 1.
   // Linear formula: z_i = z0 + (i*h) * (z1-z0), 0 <= i <= n-1
   // \sum_{i=0}^{nzones-1} z_i = n * z0 + h * (z1-z0) * nz * (nz-1) / 2

   const double slinear = n * (s0 + (h * (s1 - s0) * 0.5 * (n-1)));

   MFEM_VERIFY(std::abs(slinear - 1.0) > 1.0e-8, "Bell distribution is too "
               << "close to linear.");

   const double u = slinear < 1.0 ? 1.0 : -1.0;

   double c = 0.3;  // Initial guess

   // Newton iterations
   constexpr int maxIter = 10;
   constexpr double convTol = 1.0e-8;
   bool converged = false;
   for (int iter=0; iter<maxIter; ++iter)
   {
      const double c2 = c * c;

      const double m = 0.5 * (1.0 - (u * c2 * lnz01));
      const double dmdc = -u * c * lnz01;

      double r = 0.0;  // Residual
      double drdc = 0.0;  // Derivative of residual

      for (int i=0; i<n; ++i)
      {
         const double x = i * h;
         const double ti = exp((-(x * x) + (2.0 * x * m)) * u / c2); // Gaussian
         r += ti;

         // Derivative of Gaussian
         drdc += ((-2.0 * (-(x * x) + (2.0 * x * m)) / (c2 * c)) +
                  ((2.0 * x * dmdc) / c2)) * ti;
      }

      r *= s0;
      r -= 1.0;  // Sum of spacings should equal 1.

      if (std::abs(r) < convTol)
      {
         converged = true;
         break;
      }

      drdc *= s0 * u;

      // Newton update is -r / drdc, limited by factors of 1/2 and 2.
      double dc = std::max(-r / drdc, -0.5*c);
      dc = std::min(dc, 2.0*c);

      c += dc;
   }

   MFEM_VERIFY(converged, "Convergence failure in GaussianSpacingFunction");

   const double c2 = c * c;
   const double m = 0.5 * (1.0 - (u * c2 * lnz01));
   const double q = s0 * exp(u*m*m / c2);

   for (int i=0; i<n; ++i)
   {
      const double x = (i * h) - m;
      s[i] = q * exp(-u*x*x / c2);
   }
}

void LogarithmicSpacingFunction::CalculateSpacing()
{
   MFEM_VERIFY(n > 0 && logBase > 1.0,
               "Invalid parameters in LogarithmicSpacingFunction");

   if (sym) { CalculateSymmetric(); }
   else { CalculateNonsymmetric(); }
}

void LogarithmicSpacingFunction::CalculateSymmetric()
{
   s.SetSize(n);

   const bool odd = (n % 2 == 1);

   const int M0 = n / 2;
   const int M = odd ? (M0 + 1) : M0;

   const double h = 1.0 / ((double) M);

   double p = 1.0;  // Initialize at right endpoint of [0,1].

   for (int i=M-2; i>=0; --i)
   {
      const double p_i = (pow(logBase, (i+1)*h) - 1.0) / (logBase - 1.0);
      s[i+1] = p - p_i;
      p = p_i;
   }

   s[0] = p;

   // Even case for spacing: [s[0], ..., s[M-1], s[M-1], s[M-2], ..., s[0]]
   //   covers interval [0,2]
   // Odd case for spacing: [s[0], ..., s[M-1], s[M-2], ..., s[0]]
   //   covers interval [0,2-s[M-1]]

   const double t = odd ? 1.0 / (2.0 - s[M-1]) : 0.5;

   for (int i=0; i<M; ++i)
   {
      s[i] *= t;

      if (i < (M-1) || !odd)
      {
         s[n - i - 1] = s[i];
      }
   }
}

void LogarithmicSpacingFunction::CalculateNonsymmetric()
{
   s.SetSize(n);

   const double h = 1.0 / ((double) n);

   double p = 1.0;  // Initialize at right endpoint of [0,1].

   for (int i=n-2; i>=0; --i)
   {
      const double p_i = (pow(logBase, (i+1)*h) - 1.0) / (logBase - 1.0);
      s[i+1] = p - p_i;
      p = p_i;
   }

   s[0] = p;
}

}