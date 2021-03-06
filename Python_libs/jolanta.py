import numpy as np
import scipy.special

"""
Collects functions defining and evaluating the Jolanta model potential
"""


def Jolanta_1D(x, a=0.2, b=0.0, c=0.14):
    """
    default 1D potential; has a resonance just below 7 eV
    use for DVRs
    """
    return (a*x*x-b)*np.exp(-c*x*x)


def Jolanta_3D(r, a=0.1, b=1.2, c=0.1, l=1, as_parts=False):
    """
    default 3D potential; has a resonance at 1.75 eV - 0.2i eV
    use for DVRs
    """
    if as_parts:
        Va = a*r**2*np.exp(-c*r**2)
        Vb = b*np.exp(-c*r**2)
        Vc = 0.5*l*(l+1)/r**2
        return (Va, Vb, Vc)
    else:
        return (a*r**2-b)*np.exp(-c*r**2) + 0.5*l*(l+1)/r**2


def eval_ugto(r, a, l):
    """ 
    ununormalized GTO(r) = r^l * exp(-a*r^2) 
    """
    return r**l * np.exp(-a*r*r)


def eval_aos(Rs, alphas, Ns, l):
    """
    evaluate all AOs (all GTOs) defined by exponents alpha and angular momentum l
    Rs np.array with positions to evaluate the MOs at
    Cs[:] np.array with coefficients pertaining to normalized GTOs
    alphas GTO exponents
    Ns normalization factors
    l angular momentum common to all GTOs 
    AO_i(r_k) = N_i * r_k**l * np.exp(-alpha_i*r_k*r_k)
    """
    nAOs = len(alphas)
    nRs = len(Rs)
    AOs = np.zeros((nRs, nAOs))
    for i in range(nAOs):
        AOs[:,i] = Ns[i] * eval_ugto(Rs, alphas[i], l)
    return AOs



def eval_mos(Rs, Cs, alphas, Ns, l):
    """
    evaluate MOs defined by coefficients in Cs at all positions in Rs
    each AO is defined by:  AO(r) = r^l * exp(-a*r^2)
    MO[R_k,j] = Sum AO[R_k,i]*Cs[i,j] 
    """
    #nRs = len(Rs)
    #if len(Cs.shape) == 1:
    #    nMOs = 1
    #else:
    #    nMOs = Cs.shape[1]
    #MOs = np.zeros((nRs, nMOs))

    AOs = eval_aos(Rs, alphas, Ns, l)
    return np.dot(AOs, Cs)




def sp_gamma(a, z):
    """ 
    unnormalized upper incomplete gamma function:  integral(t^(a-1)*exp(-t),t,z,oo) 
    needed for evaluating CAP integrals analytically
    """
    return scipy.special.gamma(a) * scipy.special.gammaincc(a,z)


def gto_integral_0_inf(a, n=0):
    """ 
    integral(x^n * exp(-a*x^2), x, 0, oo) 
    where n is an integer >=0
    """
    return 0.5 * a**(-0.5*(n+1)) * scipy.special.gamma(0.5*(n+1))


def gto_integral_d_inf(a, n=0, d=1.0):
    """ 
    integral(x^n * exp(-a*x^2), x, d, oo) 
    can be directly used to evaluate CAP integrals
    W12 = <gto_1 | (r-d)^2 |gto_2> = gto_integral_d_inf(a1+a2, n=l1+l2+2, d=d)
    """
    ad2=a*d*d
    if n%2 == 0:
        m = n/2
        return 0.5 * a**(-m-0.5) * sp_gamma(m+0.5,ad2)
    else:
        m = (n+1)/2
        return 0.5 * a**(-m) * sp_gamma(m,ad2)

    
def S_12(a1, a2, l1, l2):
    """
    overlap matrix
    S12 = <gto_1 | gto_2>
    """
    return gto_integral_0_inf(a1+a2, n=l1+l2)

    
def T_12(a1, a2, l1, l2, mass=1):
    """
    kinetic energy
    T12 = 1/(2*mass) <d/dr gto_1 | d/dr gto_2>    
    assume that both l1 and l2 >= 1
    because psi(0) = 0 
    """ 
    T1 = gto_integral_0_inf(a1+a2, n=l1+l2-2)
    T2 = gto_integral_0_inf(a1+a2, n=l1+l2)
    T3 = gto_integral_0_inf(a1+a2, n=l1+l2+2)
    return 0.5*(l1*l2*T1 - 2*(l1*a2+l2*a1)*T2 + 4*a1*a2*T3) / mass


def V_12(a1, a2, l1, l2, a=0.1, b=1.2, c=0.1, l=1):
    """ 
    Jolanta_3D matrix element <gto_1| Jolanta_3D |gto_2> 
    a1, a2 are the GTO exponents, l1 and l2 the powers of r
    the unnormalized integral is returned
    (a*r**2-b)*np.exp(-c*r**2) + 0.5*l*(l+1)/r**2
    """
    Va = gto_integral_0_inf(a1+c+a2, n=l1+l2+2)*a
    Vb = gto_integral_0_inf(a1+c+a2, n=l1+l2  )*b
    Vc = gto_integral_0_inf(a1  +a2, n=l1+l2-2)*0.5*l*(l+1)
    return Va-Vb+Vc
