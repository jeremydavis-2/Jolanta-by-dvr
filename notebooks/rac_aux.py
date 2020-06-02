""" 

Here we collect only those functions needed
scipy.optimize.least_squares() based minimization


the RAC-models fit negative energies E depending on a 
strength parameter lambda: E(lambda)
    
E is is written as E = -k**2 and the model 
actually used is lambda(k)

the data to fit are passed as arrays:
k, ksq = k**2, lbs of length M
(this way, k**2 is computed only once)

each model is a Pade or rational function approximation
pade_31 implies a polynomial of third order devided 
by a polynomial of first order

Each polynomial is parametrized in a highly specialized way
motivated by quantum scattering theory. 
-> Fewer parameters than general Pade appromimants.
-> More complicated formulas.
-> All parameters are positive. 

To fit the model, minimize chi**2 = 1/M sum_i (rac-31(k_i)-lb_i)**2 

This can be done by a minimizer or by non-linear least_squares

- least_squares seems superior regarding gradient evaluations (May 2020)
- minimize and least_squares need quite different interfaces and functions
- for minimize a hand-coded chi**2 function and its gradient is required
- as gradients we need
  for minimize the vector d(chi**2)d(parameter[j])
  for least_squared the matrix d(model(k[i])-lambda[i])/d(parameter[j]) 
- minimize takes one function that returns f, and grad f, least_squares doesn't

- least_squares seems rock stable for rac-42, too

"""

import numpy as np
from scipy.optimize import curve_fit

def res_ene(alpha, beta):
    """ resonance energy Eres = Er - i*Gamma/2 from alpha and beta """
    Er = beta**2 - alpha**4
    G = 4*alpha**2 * abs(beta)
    return Er, G

def guess(Er, G):
    """ inverse of res_ene
        intented for computing a guess for alpha and beta from a guess for Eres """
    ag=0.5*np.sqrt(2.0)*(-2*Er + np.sqrt(4*Er**2 + G**2))**0.25
    bg=0.5*G/np.sqrt(-2*Er + np.sqrt(4*Er**2 + G**2))
    return [ag, bg]

def linear_extra(ls, Es):
    """ 
    find E(l=0) from an f(l)=m*l+b model 
    used to find start parameters
    """
    def f(x, m, b):
        return m*x+b
    popt, pcov = curve_fit(f, ls, Es)
    return f(0,popt[0],popt[1])

def chi2_gen(params, ks, k2s, lbs, pade):
    """
    chi2 = mean of squared deviations
    passed to basin_hopping()
    the least_squares() wrapper function needs to return 2*res.cost/len(ks)
    """
    diffs = pade(params, ks, k2s, lbs)
    return np.sum(np.square(diffs)) / len(ks)

def pade_gen_j_lsq(params, ks, k2s, lbs, pade_lsq, step=1e-5, tiny=1e-8):
    """
    for testing the pade_j_lsq() functions used in least_squares() setups
    never used in production runs, rather use the interal gradient
    """
    n_kappa = len(ks)
    n_param = len(params)
    p0 = list(params)
    dfs = np.zeros((n_param,n_kappa))
    for ip in range(n_param):
        h = step*params[ip] + tiny
        pp = np.array(p0[:ip] + [p0[ip]+h] + p0[ip+1:])
        pm = np.array(p0[:ip] + [p0[ip]-h] + p0[ip+1:])
        dp = pade_lsq(pp, ks, k2s, lbs)
        dm = pade_lsq(pm, ks, k2s, lbs)
        dfs[ip,:] = (dp-dm)/(2*h)
    return np.transpose(dfs)

def pade_31_lsq(params, k, ksq, lmbda):
    """
    model to fit f(k[i]) to lmbda[i]
    ksq = k**2 is computed only once
    params: [lambda0, alpha, beta, delta]
    returns model(k) - lbs
    For details see DOI: 10.1140/epjd/e2016-70133-6
    """
    l0, a, b, d = params
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (ksq + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2
    rac31 = l0 * num / den
    return rac31 - lmbda

def pade_31j_lsq(params, k, ksq, lbs):
    """
    'jac' for pade_31_lsq
    arguments must be identical with pade_lsq()
    computes the matrix del pade(k[i])/del para[j] 
    returns the M-by-N matrix needed by scipy.optimize.least_squares
    M = number of data points
    N = number of parameters
    least_squares() needs the transpose 
    """
    l, a, b, d = params
    a2, b2, d2 = a*a, b*b, d*d
    a4b2 = a2*a2 + b2
    aak2 = a2*k*2
    ddk = d2*k
    fr1 = (ksq + aak2 + a4b2)
    fr2 = (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2
    dl = fr1*fr2/den
    da = -4*a*ksq*l * fr2 * (a2*a2*d2 + a2*fr2 - b2*d2 + k) / den**2
    db = -2*b*ksq*l * fr2 * (2*a2*d2 + fr2) / den**2
    dd = 4*a2*d*ksq*l * fr1/den**2
    return np.transpose(np.array([dl, da, db, dd]))

def pade_42_lsq(params, k, ksq, lmbda):
    """
    model to fit f(k[i]) to lmbda[i]
    ksq = k**2 is computed only once
    params: [lambda0, alpha, beta, gamma, delta, omega]
    returns f(k) - lbs
    For details see DOI: 10.1140/epjd/e2016-70133-6
    Roman's new factorization:
    divide the second factor by a**4 + b**2 so that is becomes 1 + g**2*k + d**2*k**2
    with d**2=1/(a**4+b**2)  and g**2=2*a**2/(a**4+b**2)
    """
    l0, a, b, g, d, o = params
    A, B, G, D, O = a**2, b**2, g**2, d**2, o**2
    TA = 2*A
    A2B = A*A + B
    C = TA + G*A2B
    f1 = ksq + TA*k + A2B
    f2 = 1 + G*k + D*ksq
    den = A2B + C*k + O*ksq
    f = l0 * f1 * f2 / den
    return f - lmbda

def pade_42j_lsq(params, k, ksq, lmbda):
    """
    'jac' for pade_42_lsq
    arguments must be identical with pade_lsq()
    computes the matrix del pade(k[i])/del para[j] 
    returns the M-by-N matrix needed by scipy.optimize.least_squares
    M = number of data points
    N = number of parameters
    least_squares() needs the transpose 
    """
    l0, a, b, g, d, o = params
    A, B, G, D, O = a**2, b**2, g**2, d**2, o**2
    TA = 2*A
    A2B = A*A + B
    C = TA + G*A2B
    f1 = ksq + TA*k + A2B
    f2 = 1 + G*k + D*ksq
    den = A2B + C*k + O*ksq
    dl0 = f1 * f2 / den
    da = -4*a*ksq*l0 * f2 * (A*A*G - A*O + A - B*G + (A*G - O + 1)*k) / den**2
    db = -2*b*ksq*l0 * f2 * (TA*G -O + 1 + G*k) / den**2
    dg = -2*g*ksq*l0 * f1 * ((A*A*D + B*D - O)*k - TA) / den**2
    dd = 2*d*ksq*l0 * f1 / den
    do = -2*o*ksq*l0 * f1*f2 / den**2
    return np.transpose(np.array([dl0, da, db, dg, dd, do]))


def pade_53_lsq(params, k, ksq, lmbda):
    """
    model to fit f(k[i]) to lmbda[i]
    ksq = k**2 is computed only once
    params: [lambda0, alpha, beta, gamma, delta, omega]
    returns f(k) - lbs
    For details of the basic method see DOI: 10.1140/epjd/e2016-70133-6
    Roman's new factorization:
    divide the second factor by a**4 + b**2 so that is becomes 1 + g**2*k + d**2*k**2
    with d**2=1/(a**4+b**2)  and g**2=2*a**2/(a**4+b**2)
    """
    l0, a, b, g, d, e, o, r = params
    A, B, G, D, E, O, R = a**2, b**2, g**2, d**2, e**2, o**2, r**2
    TA = 2*A
    A2B = A*A + B
    C = TA + (G+E)*A2B
    f1 = ksq + TA*k + A2B
    f2 = 1 + G*k + D*ksq
    f3 = 1 + E*k
    den = A2B + C*k + O*ksq + R*k*ksq
    f = l0 * f1 * f2 *f3 / den
    return f - lmbda

def pade_53j_lsq(params, k, ksq, lmbda):
    """
    'jac' for pade_53_lsq
    arguments must be identical with pade_lsq()
    computes the matrix del pade(k[i])/del para[j] 
    returns the M-by-N matrix needed by scipy.optimize.least_squares
    M = number of data points
    N = number of parameters
    least_squares() needs the transpose 
    """
    l0, a, b, g, d, e, o, r = params
    A, B, G, D, E, O, R = a**2, b**2, g**2, d**2, e**2, o**2, r**2
    TA = 2*A
    A2B = A*A + B
    C = TA + (G+E)*A2B
    f1 = ksq + TA*k + A2B
    f2 = 1 + G*k + D*ksq
    f3 = 1 + E*k
    den = A2B + C*k + O*ksq + R*k*ksq

    dl0 = f1 * f2 * f3 / den
    
    da = 4*a*ksq*l0 * f2 * f3 * (-A*A*(E+G) + A*O - A + B*(E+G) + (A*(R-E-G) + O - 1)*k + R*ksq) / den**2

    db = 2*b*ksq*l0 * f2 * f3 * (-TA*(E+G) + O - 1 + (R-E-G)*k) / den**2
    
    dg = 2*g*ksq*l0 * f1 * f3 * (TA + A2B*E + (O - A2B*D)*k + R*ksq) / den**2

    dd = 2*d*ksq*l0 * f1 * f3 / den

    de = 2*e*ksq*l0 * f1 * f2 * (A2B*G + TA + O*k + R*ksq) / den**2

    do = -2*o*ksq*l0 * f1 * f2 * f3 / den**2

    dr = -2*r*ksq*k*l0 * f1 * f2 * f3 / den**2

    return np.transpose(np.array([dl0, da, db, dg, dd, de, do, dr]))
