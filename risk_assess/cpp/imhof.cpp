// Title: Imhof (1961) algorithm
// Ref. (book or article): {J. P. Imhof, Computing the Distribution of Quadratic Forms in Normal Variables, Biometrika, Volume 48, Issue 3/4 (Dec., 1961), 419-426

// Description:
// Distribution function (survival function in fact) of quadratic forms in normal variables using Imhof's method.

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

using std::atan;
using std::exp;
using std::sin;
using std::pow;

extern "C" {
  
  struct problem_params {
    double x;
    int lambdalen;
    double *lambda;
    double *h;
    double *delta;
  };
  
  /** This is the function under the integral sign in equation (3.2), Imhof (1961), p.422
   * u: mesh point
   * lambda
   * lambdalen
   * h
   * x : Prob(Q>x)
   * */
  inline double imhoffunc(const double &u, double *lambda, const int &lambdalen, double *h, const double &x, double *delta2)
    
  { double theta = 0.0;
    double rho = 1.0;
    double pow_lambda_u; // Repeated expression.
    for (int i = 0; i <= lambdalen-1; i++){
      pow_lambda_u = pow(lambda[i] * u, 2.0);
      theta+=h[i] * std::atan(lambda[i] * u) + delta2[i] * lambda[i] * u / (1.0 + pow_lambda_u);
      rho*=pow(1.0 + pow_lambda_u, 0.25 * h[i]) * exp(0.5 * delta2[i] * pow_lambda_u / (1.0 + pow_lambda_u));
    }
    theta =  0.5 * theta - 0.5 * x * u;
    return (std::sin(theta)) / (u * rho);
  }



  double f(double u, void *ex) 

  { 
    struct problem_params * inputs = (struct problem_params *) ex;
    return imhoffunc(u, inputs->lambda, inputs->lambdalen, inputs->h, inputs->x, inputs->delta);
  }
  

  void upper_tail_prob(const double x, double *lambda, const int lambdalen, double *h, double *delta2, double *Qx,
                 double *epsabs_out, const double epsabs, const double epsrel, const int limit)
  {
    //
    // Parsing inputs.
    //
    struct problem_params ex = {
      .x = x,
      .lambdalen = lambdalen,
      .lambda = lambda,
      .h = h,
      .delta = delta2
    };

    //
    // Outputs
    // 
    // resulting integral value and absolute error.
    double result, abserr;

    // Define the problem in GSL.
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(limit);
    gsl_function F;
    F.function = &f;
    F.params = &ex;

    // Integrate F from 0 to infinity, equation 3.2 of the Imhof paper.
    gsl_set_error_handler_off();
    gsl_integration_qagiu(&F, 0.0, epsabs, epsrel, limit, w, &result, &abserr);
    gsl_integration_workspace_free(w);


    // Output probability, Qx, and absolute error.
    Qx[0] = 0.5 + result / M_PI;
    epsabs_out[0] = abserr;

    return;
  }
  
}


// int main(){


//   // Prob(Q>t)
//   constexpr double t = 1.0;
//   constexpr int lambdalen = 2;
//   double lambda[lambdalen] = {0.9, 1.0};
//   double h[lambdalen] = {1.0, 1.0};
//   double delta[lambdalen] = {2.25, 2.89};
//   double *Qx; Qx = new double[1];
//   double *epsabs_out; epsabs_out = new double[1];
//   constexpr double epsabs = 5e-5;
//   constexpr double epsrel = 5e-5;
//   constexpr int limit = 10000;

//   constexpr int iters = 10000;
//   auto start = high_resolution_clock::now(); 
//   for (int i = 0; i < iters; i++){
//     upper_tail_prob(t, lambda, lambdalen, h, delta, Qx, epsabs_out, epsabs, epsrel, limit);
//   }
//   auto stop = high_resolution_clock::now(); 
//   auto duration = duration_cast<microseconds>(stop - start); 
//   std::cout << "Time taken by function: "
//          << duration.count()/iters << " microseconds" << std::endl; 
//   std::cout<<"Value: "<< Qx[0] <<std::endl;
//   return 0;
// }