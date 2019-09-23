import sympy as sp
import numpy as np
from functools import reduce
from itertools import accumulate
import time

def compute_all_thetas(theta0, utheta):
    return theta0 + np.cumsum(utheta)

def compute_all_vs(v0, accels_uncertain, dt):
    return v0 + dt*np.cumsum(accels_uncertain)

def compute_beta2_mono_moments(alpha, beta, monos):
    max_order_needed = max(max(monos))
    #assume iid beta moments
    beta2_moments = compute_beta2_moments(alpha,beta,max_order_needed)
    cross_moments = [reduce(lambda prev,ni: prev*beta2_moments[ni],[1] + list(tup)) for tup in monos]
    return cross_moments

def compute_beta_moments(alpha,beta,order):
    #Compute beta moments up to the given order
    #the returned list indices should match the moment orders
    #e.g. the return[i] should be the ith beta moment
    fs = map(lambda r: (alpha + r)/(alpha + beta + r), range(order))
    return [1] + list(accumulate(fs, lambda prev,n: prev*n))

def compute_beta2_moments(alpha, beta, n):
    beta_moments = compute_beta_moments(alpha, beta, n)
    return [beta_moments[i]*2**i for i in range(len(beta_moments))]

start_t = time.time()
# Problem Parameters
dt = 0.05

# Initial State
x0 = -1.5
y0 = 0.0
theta0 = 0.0
v0 = 0.0

#Define the polynomials
p = lambda x,y: -2*x**2 - 4*y**2 + 3*y**4 + 2.0*x**4 - 0.1

p_poly = lambda x,y,w: sp.poly(p(x,y), w)

p_poly_sq = lambda x,y,w: p_poly(x,y,w)**2

n_t = 4 #number of control inputs, so we will have n_t + 1 states

u_accel = sp.symbols('ua0:' + str(n_t),real = True, constant = True)
#u_steer = sp.symbols('us0:' + str(n_t), real = True, constant = True)
w_accel = sp.symbols('w0:' + str(n_t), real = True, constant = True)

#thetas = compute_all_thetas(theta0, u_steer)


thetas = sp.symbols('thetas0:' + str(n_t), real = True, constant = True)

accel_uncertain = np.asarray(list(map(lambda i: w_accel[i]*u_accel[i], range(n_t))))
vs_uncertain_dt = compute_all_vs(v0, accel_uncertain, dt)

cos_thetas = [sp.cos(t) for t in thetas]
sin_thetas = [sp.sin(t) for t in thetas]

xs = list(accumulate([x0] + [x for x in range(n_t)], lambda pre,i: pre + vs_uncertain_dt[i]*cos_thetas[i]))
ys = list(accumulate([y0] + [x for x in range(n_t)], lambda pre,i: pre + vs_uncertain_dt[i]*sin_thetas[i]))

xs_final = xs[-1]
ys_final = ys[-1]

#We have our polynomials now, yay
final_p = p_poly(xs_final, ys_final, w_accel)
final_p_sq = p_poly_sq(xs_final, ys_final, w_accel)

coefs_p = np.asarray(final_p.coeffs())
monos_p = final_p.monoms()

coefs_p_sq = np.asarray(final_p_sq.coeffs())
monos_p_sq = final_p_sq.monoms()

#We will can coefs with control input data and monos with moments now!
coefs_p_lam = sp.lambdify([u_accel, thetas], coefs_p) #now a function that accepts two lists as inputs
coefs_p_sq_lam = sp.lambdify([u_accel, thetas], coefs_p_sq) #now a function that accepts two lists as inputs

end_t = time.time()
print("Total time for precomputation of the polynomial is: " + str(end_t - start_t))

start_t = time.time()
#We can use monos_p and monos_p_sq to determine the moments we need given any uncertain vector with independent elements
alpha = 500
beta = 550
mono_moments_p = np.asarray(compute_beta2_mono_moments(alpha, beta, monos_p)) #these moments correspond to the monos
mono_moments_p_sq = np.asarray(compute_beta2_mono_moments(alpha, beta, monos_p_sq)) #these moments correspond to the monos

#Generate simple control sequence
ua_vals = np.ones((1,n_t)).ravel().tolist()
theta_vals = np.zeros((1,n_t)).ravel().tolist()

#Evaluate coefficients
coefs_p_eval = coefs_p_lam(ua_vals, theta_vals)
coefs_p_sq_eval = coefs_p_sq_lam(ua_vals, theta_vals)

#Evaluate first and second moments of p
p_mean = np.dot(coefs_p_eval, mono_moments_p)
p_second_moment = np.dot(coefs_p_sq_eval, mono_moments_p_sq)

variance = p_second_moment - p_mean**2
end_t = time.time()
print("Total time for computation is: " + str(end_t - start_t))
print("p mean is: " + str(p_mean))
print('p variance is: ' + str(variance))
