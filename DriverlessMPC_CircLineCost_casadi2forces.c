/* 
 * CasADi to FORCES Template - missing information to be filled in by createCasadi.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-19. All rights reserved.
 *
 * This file is part of the FORCES client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif
    
#include "DriverlessMPC_CircLineCost/include/DriverlessMPC_CircLineCost.h"    
    
/* prototyes for models */
extern void DriverlessMPC_CircLineCost_model_1(const DriverlessMPC_CircLineCost_float **arg, DriverlessMPC_CircLineCost_float **res);
extern void DriverlessMPC_CircLineCost_model_1_sparsity(solver_int32_default i, solver_int32_default *nrow, solver_int32_default *ncol, const solver_int32_default **colind, const solver_int32_default **row);
extern void DriverlessMPC_CircLineCost_model_50(const DriverlessMPC_CircLineCost_float **arg, DriverlessMPC_CircLineCost_float **res);
extern void DriverlessMPC_CircLineCost_model_50_sparsity(solver_int32_default i, solver_int32_default *nrow, solver_int32_default *ncol, const solver_int32_default **colind, const solver_int32_default **row);
    

/* copies data from sparse matrix into a dense one */
static void sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, DriverlessMPC_CircLineCost_float *data, DriverlessMPC_CircLineCost_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for( j=colidx[i]; j < colidx[i+1]; j++ )
        {
            out[i*nrow + row[j]] = data[j];
        }
    }
}

/* CasADi - FORCES interface */
extern void DriverlessMPC_CircLineCost_casadi2forces(DriverlessMPC_CircLineCost_float *x,        /* primal vars                                         */
                                 DriverlessMPC_CircLineCost_float *y,        /* eq. constraint multiplers                           */
                                 DriverlessMPC_CircLineCost_float *l,        /* ineq. constraint multipliers                        */
                                 DriverlessMPC_CircLineCost_float *p,        /* parameters                                          */
                                 DriverlessMPC_CircLineCost_float *f,        /* objective function (scalar)                         */
                                 DriverlessMPC_CircLineCost_float *nabla_f,  /* gradient of objective function                      */
                                 DriverlessMPC_CircLineCost_float *c,        /* dynamics                                            */
                                 DriverlessMPC_CircLineCost_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 DriverlessMPC_CircLineCost_float *h,        /* inequality constraints                              */
                                 DriverlessMPC_CircLineCost_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 DriverlessMPC_CircLineCost_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                            */
								 solver_int32_default iteration /* iteration number of solver                          */)
{
    /* CasADi input and output arrays */
    const DriverlessMPC_CircLineCost_float *in[4];
    DriverlessMPC_CircLineCost_float *out[7];
    
    /* temporary storage for casadi sparse output */
    DriverlessMPC_CircLineCost_float this_f;
    DriverlessMPC_CircLineCost_float nabla_f_sparse[5];
    
    
    DriverlessMPC_CircLineCost_float c_sparse[4];
    DriverlessMPC_CircLineCost_float nabla_c_sparse[16];
            
    
    /* pointers to row and column info for 
     * column compressed format used by CasADi */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for CasADi */
    in[0] = x;
    in[1] = p; /* maybe should be made conditional */
    in[2] = l; /* maybe should be made conditional */     
    in[3] = y; /* maybe should be made conditional */
    
    /* set outputs for CasADi */
    out[0] = &this_f;
    out[1] = nabla_f_sparse;
                
	 if ((stage >= 0 && stage < 49))
	 {
		 /* set inputs */
		 out[2] = c_sparse;
		 out[3] = nabla_c_sparse;
		 /* call CasADi */
		 DriverlessMPC_CircLineCost_model_1(in, out);

		 /* copy to dense */
		 if( nabla_f )
		 {
			 DriverlessMPC_CircLineCost_model_1_sparsity(3, &nrow, &ncol, &colind, &row);
			 sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		 }
		 if( c )
		 {
		 DriverlessMPC_CircLineCost_model_1_sparsity(4, &nrow, &ncol, &colind, &row);
		 sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		 }
		 if( nabla_c )
		 {
			 DriverlessMPC_CircLineCost_model_1_sparsity(5, &nrow, &ncol, &colind, &row);
			 sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		 }
		 
	 }

	 if ((stage >= 49 && stage < 50))
	 {
		 /* call CasADi */
		 DriverlessMPC_CircLineCost_model_50(in, out);

		 /* copy to dense */
		 if( nabla_f )
		 {
			 DriverlessMPC_CircLineCost_model_50_sparsity(3, &nrow, &ncol, &colind, &row);
			 sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		 }
		 
	 }

         
    
    /* add to objective */
    if( f )
    {
        *f += this_f;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif