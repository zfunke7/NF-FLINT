# import os
# os.chdir('C:/Users/beler/Documents/Doctorate/ASEN 6060 - Advanced Astrodynamics/Homeworks')
# from CR3BP import *
# os.chdir('C:/Users/beler/Documents/Doctorate/ASEN 6014 - Formation Flying/Homework/')
# from utils_6014 import *
# os.chdir('C:/Users/beler/Documents/Doctorate/ASEN 6014 - Formation Flying/Project 1/')
# import tracemalloc
import math
import numpy as np
# import orjson
import scipy.special
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from time import time
import pickle as pkl
import matplotlib.pyplot as plt

import flint
from flint import fmpq, fmpq_mpoly_ctx
mu_EM=0.012150584394709708


''' Brevity & Helper Functions '''
fac = scipy.special.factorial
gam = scipy.special.gamma
nm = np.linalg.norm

def q2float(q):
    ''' Evaluates a rational number of FLINT type fmpq into a Python float. '''
    return int(q.numer()) / int(q.denom())


def float2q(f):
    ''' Converts a Python float into a rational number of FLINT type fmpq. '''
    return fmpq(*f.as_integer_ratio())

''' The idea of this module is to use FLINT, wrapped via python-flint, which is a set
    of highly optimized C routines for manipulating multivariate polynomials. As such, 
    this module dispenses with the JAX dependency and uses numpy throughout, however, 
    python-flint (v0.7.0 used here) is probably a riskier dependency than JAX. 
    FLINT does not provide multivariate polynomial support for arbitrary coefficient 
    domains (i.e., the ring of complex numbers), so this module implements its own class
    (Poly) to compose all necessary operations (addition, multiplication, differentiation)
    using only flint.fmpq_mpoly, which constructs polynomial representations with coefficients
    from the ring of rational numbers. The cost: polynomials are doubled in size because
    the complex number "j" must now be represented as a 7th variable. Nevertheless, polynomial
    operations are orders of magnitudes faster than the JAX/Python implementation with dicts. '''


class Poly:
    ''' Wraps a consistent representation between Python dicts ~{6-tuple, coeff} and python-flint fmpq_mpoly objects with
        7-tuple coefficients (coeff at index 0 is the imag # "j"). Addition, multiplication, and derivatives are performed with FLINT. '''

    def __init__(self, *args):
        if type(args[0]) == dict:
            self.d = args[0]
            self.ctx = fmpq_mpoly_ctx.get(('j', 'q1', 'q2', 'q3', 'p1', 'p2', 'p3'))
            self.poly = self.dict2poly(self.d)
        elif type(args[0]) == flint.types.fmpq_mpoly.fmpq_mpoly:
            self.poly = args[0]
            self.ctx = self.poly.context()
        elif type(args[0]) in [Poly, RealPoly]:
            self = args[0]

    def dict2poly(self, d):
        if type([dv for dv in d.values()][0]) == Poly: # input is 'by degree'
            deg_polys = [dv for dv in d.values()]
            out = deg_polys[0]
            for dp in deg_polys[1:]:
                out += dp
            return out.poly
        else: # input is a dict with degree keys and float/complex values
            keys = [tuple([0, *key]) for key in d.keys()]
            imag_keys = [tuple([1, *key]) for key in d.keys()]
            keys.extend(imag_keys)
            coeffs = [fmpq(*float(np.real(val)).as_integer_ratio()) for val in d.values()]
            imag_coeffs = [fmpq(*float(np.imag(val)).as_integer_ratio()) for val in d.values()]
            coeffs.extend(imag_coeffs)
            d_ext = dict(zip(keys, coeffs))
            return self.ctx.from_dict(d_ext)

    def as_dict(self, real=False):
        try:
            return self.d
        except:
            d0 = self.poly.to_dict()
            unique_keys = list(set(key[1:] for key in d0))
            df = dict(zip(unique_keys, np.zeros(len(unique_keys))))
            for key in d0.keys():
                coeff = q2float(d0[key]) * (1j)**(int(key[0]))
                df[key[1:]] += coeff
            for key in d0.keys():
                df[key[1:]] *= int(not np.isclose(df[key[1:]],0))
                if real:
                    df[key[1:]] = np.real(df[key[1:]])
            self.d = df
            return df

    def as_array(self):
        try:
            d0 = self.d
        except AttributeError:
            d0 = self.as_dict()
        a = np.zeros((len(d0), 7), dtype=complex)
        for i, key in enumerate(d0.keys()):
            a[i, 0] = d0[key]
            a[i, 1:] = [int(k) for k in key]
        self.a = a
        return a

    def simplify(self, real=False): # Removes terms that sum to 0, accounting for complex sums
        try:
            del self.d # Force re-processing by as_dict()
        except AttributeError:
            pass
        d = self.as_dict(real=real)
        if real:
            return RealPoly(self.dict2poly(d))
        else:
            return Poly(self.dict2poly(d))

    def by_degree(self):
        d0 = self.poly.to_dict()
        unique_degs = list(set([sum(key[1:]) for key in d0.keys()]))
        df = dict(zip(unique_degs, [None for i in range(len(unique_degs))]))
        for deg in unique_degs:
            deg_dict = dict(zip([key for key in d0.keys() if sum(key[1:]) == deg],
                                [d0[key] for key in d0.keys() if sum(key[1:]) == deg]))
            df[deg] = Poly(self.ctx.from_dict(deg_dict))
        return df

    def transform(self, M):
        g0 = self.ctx.gens()
        j_ = g0[0]
        sub = [sum([fmpq(*(np.real(M[k,i]).as_integer_ratio()))*g0[i+1] + \
                    fmpq(*(np.imag(M[k,i]).as_integer_ratio()))*g0[i+1]*j_ \
                    for i in range(6)]) for k in range(6)]
        return Poly(self.poly.compose(j_, *sub))

    def eval(self, X, array=False):
        if array:
            try:
                a = self.a
            except AttributeError:
                a = self.as_array()
            return np.sum(a[:,0].T * np.prod((X ** a[:,1:]), axis=1))
        else:
            j_ = self.ctx.gens()[0]
            sub = [fmpq(*float(np.real(x)).as_integer_ratio()) + fmpq(*float(np.imag(x)).as_integer_ratio()) * j_ \
                   for x in X]
            pre_eval = list(self.poly.compose(j_, *sub).terms())
            return sum([int(term[1].numer()) / int(term[1].denom()) * (1j) ** (int(term[0][0])) for term in pre_eval])

    def deriv(self, var):
        return Poly(self.poly.derivative(var+1))

    def __add__(self, other):
        return Poly(self.poly + other.poly)

    def __sub__(self, other):
        return Poly(self.poly - other.poly)

    def __mul__(self, other):
        if type(other) == Poly:
            return Poly(self.poly * other.poly)
        elif type(other) in (float, int):
            return Poly(float2q(other) * self.poly)
        elif type(other) == complex:
            print('yes')
            p_other = self.ctx.from_dict({(0,0,0,0,0,0,0): float2q(np.real(other)),
                                     (1,0,0,0,0,0,0): float2q(np.imag(other))})
            return Poly(self.poly * p_other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Poly(-self.poly)


class RealPoly:
    ''' Wraps a consistent representation between Python dicts ~{6-tuple, coeff} and python-flint fmpq_mpoly objects with
        6-tuple coefficients. Addition, multiplication, and derivatives are performed with FLINT. '''

    def __init__(self, *args):
        ''' Assumes coefficients of input are strictly real. '''
        self.ctx = fmpq_mpoly_ctx.get(('q1', 'q2', 'q3', 'p1', 'p2', 'p3'))
        if type(args[0]) == dict:
            self.d = args[0]
            self.poly = self.dict2poly(self.d)
        elif type(args[0]) == flint.types.fmpq_mpoly.fmpq_mpoly:
            if len(args[0].degrees()) == 6:
                self.poly = args[0]
            elif len(args[0].degrees()) == 7:
                self.poly = self.dict2poly(Poly(args[0]).as_dict())
        elif type(args[0]) == Poly:
            self.poly = self.dict2poly(args[0].as_dict())
        elif type(args[0]) == RealPoly:
            self = args[0]

    def dict2poly(self, d):
        if type([dv for dv in d.values()][0]) == Poly: # input is 'by degree'
            deg_polys = [dv for dv in d.values()]
            out = RealPoly(deg_polys[0])
            for dp in deg_polys[1:]:
                out += RealPoly(dp)
            return out.poly
        else: # input is a dict with degree keys and float/complex values
            keys = [key[-6:] for key in d.keys()]
            coeffs = [fmpq(*float(np.real(val)).as_integer_ratio()) for val in d.values()]
            d_ext = dict(zip(keys, coeffs))
            return self.ctx.from_dict(d_ext)

    def as_dict(self):
        try:
            return self.d
        except AttributeError:
            d0 = self.poly.to_dict()
            df = dict(zip(d0.keys(), np.zeros(len(d0.values()))))
            for key in d0.keys():
                coeff = q2float(d0[key])
                df[key] += coeff
            self.d = df
            return df

    def as_array(self):
        try:
            d0 = self.d
        except AttributeError:
            d0 = self.as_dict()
        a = np.zeros((len(d0), 7), dtype=float)
        for i, key in enumerate(d0.keys()):
            a[i, 0] = d0[key]
            a[i, 1:] = [int(k) for k in key]
        self.a = a
        return a

    def by_degree(self):
        d0 = self.poly.to_dict()
        unique_degs = list(set([sum(key) for key in d0.keys()]))
        df = dict(zip(unique_degs, [None for i in range(len(unique_degs))]))
        for deg in unique_degs:
            deg_dict = dict(zip([key for key in d0.keys() if sum(key) == deg],
                                [d0[key] for key in d0.keys() if sum(key) == deg]))
            df[deg] = RealPoly(self.ctx.from_dict(deg_dict))
        return df

    def transform(self, M):
        g0 = self.ctx.gens()
        sub = [sum([fmpq(*(M[k,i].as_integer_ratio()))*g0[i+1]  \
                    for i in range(6)]) for k in range(6)]
        return RealPoly(self.poly.compose(*sub))

    def eval(self, X, array=False):
        if array:
            try:
                a = self.a
            except AttributeError:
                a = self.as_array()
            return np.sum(a[:,0].T * np.prod((X ** a[:,1:]), axis=1))
        else:
            out = self.poly.subs(dict(zip([str(gen) for gen in self.ctx.gens()], \
                                          [float2q(x) for x in X])))
            out = q2float(out[0, 0, 0, 0, 0, 0])
            return out

    def deriv(self, var):
        return RealPoly(self.poly.derivative(var))

    def __add__(self, other):
        if type(other) == Poly:
            return Poly(self.as_dict()) + other
        elif type(other) == RealPoly:
            return RealPoly(self.poly + other.poly)

    def __sub__(self, other):
        if type(other) == Poly:
            return Poly(self.as_dict()) - other
        elif type(other) == RealPoly:
            return RealPoly(self.poly - other.poly)

    def __mul__(self, other):
        if type(other) == Poly:
            return Poly(self.as_dict()) * other
        elif type(other) == RealPoly:
            return RealPoly(self.poly * other.poly)
        elif type(other) in (float, int):
            return RealPoly(float2q(other) * self.poly)
        elif type(other) == complex:
            p_other = self.ctx.from_dict({(0,0,0,0,0,0,0): float2q(np.real(other)),
                                     (1,0,0,0,0,0,0): float2q(np.imag(other))})
            return Poly(self.poly) * p_other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return RealPoly(-self.poly)


def csqrt(a):
    return (np.sign(a)+1)/2 * np.sqrt(abs(a)) + (1-np.sign(a))/2 * np.sqrt(abs(a))*1j

def euler_quintic(g, mu=mu_EM):
    return g**5 - (3-mu)*g**4 + (3-2*mu)*g**3 - mu*g**2 +2*mu*g-mu
# Distance from Moon to L1 point
gamma_1 = fsolve(euler_quintic, (mu_EM/3)**(1/3))[0]

''' Coordinate Transformation  Functions '''
# Synodic Cartesian -> Synodic Canonical -> Scaled/Recentered Canonical
# -> Real Diagonal Canonical -> Complex Diagonal Canonical -> Action-Angle


def cart2canon(X_in, reverse=False):
    ''' Convert Cartesian synodic coordinates to Hamiltonian coordinates/momenta. '''
    if not reverse:
        X, Y, Z, dX, dY, dZ = X_in
        PX, PY, PZ = dX - Y, dY + X, dZ
        return np.array([X, Y, Z, PX, PY, PZ])
    elif reverse:
        X, Y, Z, PX, PY, PZ = X_in
        dX, dY, dZ = PX + Y, PY - X, PZ
        return np.array([X, Y, Z, dX, dY, dZ])


def scale_recenter(Z_in, gamma=gamma_1, mu=mu_EM, reverse=False):
    ''' Recenters the canonical coordinates (q,p) about L1, normalizing by the L1-Moon distance gamma.'''
    a = 1 - mu - gamma_1
    if not reverse:
        X, Y, Z, PX, PY, PZ = Z_in
        x = (1 / gamma) * (X - a)  # Schwab
        y = (1 / gamma) * Y
        z = (1 / gamma) * Z
        # px, py, pz = PX, PY, PZ  # Do nothing with momenta
        px = (1/gamma)*PX
        py = (1/gamma)*(PY - a)
        pz = (1/gamma)*PZ
        return np.array([x, y, z, px, py, pz])
    elif reverse:
        x, y, z, px, py, pz = Z_in
        X = gamma*x + a
        Y = gamma*y
        Z = gamma*z
        # PX, PY, PZ = px, py, pz  # Do nothing with momenta
        PX = gamma*px
        PY = gamma*py + a
        PZ = gamma*pz
        return np.array([X, Y, Z, PX, PY, PZ])


def c_n(n, gamma=gamma_1, mu=mu_EM):
    ''' for L1 point only '''
    return (1 / (gamma ** 3)) * (mu + (-1) ** n * (1 - mu) * (gamma / (1 - gamma)) ** (n + 1))


def diag_linear(gamma=gamma_1, mu=mu_EM):
    '''Applies the linear symplextic transofrmation to express the state in the real normal form basis of
       the linear system. Ref: Jorba, Angel, "A Methodology for the Numerical Computation of Normal Forms..." '''
    c2 = c_n(2, gamma=gamma, mu=mu_EM)
    w2 = np.sqrt(c2)
    n1 = 0.5 * (c2 - 2 - np.sqrt(9 * c2 ** 2 - 8 * c2))
    n2 = 0.5 * (c2 - 2 + np.sqrt(9 * c2 ** 2 - 8 * c2))
    w1 = np.sqrt(-n1)
    lam = np.sqrt(n2)
    dlam = 2 * lam * ((4 + 3 * c2) * lam ** 2 + 4 + 5 * c2 - 6 * c2 ** 2)
    dw1 = w1 * ((4 + 3 * c2) * w1 ** 2 - 4 - 5 * c2 + 6 * c2 ** 2)
    s1 = np.sqrt(dlam)
    s2 = np.sqrt(dw1)
    C11 = 2 * lam / s1
    C14 = -2 * lam / s1
    C15 = 2 * w1 / s2
    C21 = (1 / s1) * (lam ** 2 - 2 * c2 - 1)
    C22 = (1 / s2) * (-w1 ** 2 - 2 * c2 - 1)
    C24 = (1 / s1) * (lam ** 2 - 2 * c2 - 1)
    C33 = 1 / np.sqrt(w2)
    C41 = (1 / s1) * (lam ** 2 + 2 * c2 + 1)
    C42 = (1 / s2) * (-w1 ** 2 + 2 * c2 + 1)
    C44 = (1 / s1) * (lam ** 2 + 2 * c2 + 1)
    C51 = (1 / s1) * (lam ** 3 + (1 - 2 * c2) * lam)
    C54 = (1 / s1) * (-lam ** 3 - (1 - 2 * c2) * lam)
    C55 = (1 / s2) * (-w1 ** 3 + (1 - 2 * c2) * w1)
    C66 = np.sqrt(w2)

    D = np.array([[C11, 0, 0, C14, C15, 0],
                   [C21, C22, 0, C24, 0, 0],
                   [0, 0, C33, 0, 0, 0],
                   [C41, C42, 0, C44, 0, 0],
                   [C51, 0, 0, C54, C55, 0],
                   [0, 0, 0, 0, 0, C66]])  # From Jorba & Masdemont 1999
    return D, lam, w1, w2


def cmplx_xfrm():
    rt2 = np.sqrt(2)
    return 1 / rt2 * np.array([[rt2, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 1j, 0],
                                [0, 0, 1, 0, 0, 1j],
                                [0, 0, 0, rt2, 0, 0],
                                [0, 1j, 0, 0, 1, 0],
                                [0, 0, 1j, 0, 0, 1]])


def diagonalize(X, reverse=False, gamma=gamma_1, mu=mu_EM):
    D, l1, w1, w2 = diag_linear(gamma=gamma, mu=mu_EM)
    return np.linalg.inv(D) @ X if not reverse else D @ X


def complexify(X, reverse=False):
    ''' Complexifies the center variables associated with the L1 Hamiltonian coordinates & momenta '''
    C = cmplx_xfrm()
    if not reverse:
        Xc = np.linalg.inv(C) @ X
        return Xc
    elif reverse:
        Xr = C @ X
        if np.isclose(np.imag(Xr), 0).all():
            return np.real(Xr)
        else:
            print('Warning! Realification failed; is there a problem with the input?')
            return Xr
    return np.linalg.inv(C) @ X if not reverse else C @ X


def quad_Ham(X_cmplx, l, w1, w2):
    ''' Returns H2, the quadratic truncation of the Hamiltonian series describing the linearized eqns of motion 
        about L1. '''  # Why is this different than H_n(2)?
    q1, q2, q3, p1, p2, p3 = X_cmplx
    return np.real(l * q1 * p1 + 1j * w1 * q2 * p2 + 1j * w2 * q3 * p3)


def canon2AA(X, reverse=False):
    ''' mapping (q,p) -> (I, Th) for L1, where q1 & p1 are saddle coords, rest are center coords.
        Specify reverse=True for the inverse mapping. '''
    if not reverse:
        q1, q2, q3, p1, p2, p3 = X
        I1 = abs(q1 * p1)
        I2 = 0.5 * (q2 ** 2 + p2 ** 2)
        I3 = 0.5 * (q3 ** 2 + p3 ** 2)
        Th1 = np.log(csqrt(q1) / csqrt(p1)) if I1 != 0 else 0
        Th2 = np.arctan2(-p2, q2) % (2*np.pi)
        Th3 = np.arctan2(-p3, q3) % (2*np.pi)
        return np.array([I1, I2, I3, Th1, Th2, Th3])
    elif reverse: # For L1 only - depends on stability types
        I1, I2, I3, Th1, Th2, Th3 = X
        q1, p1 = np.exp(Th1)*csqrt(I1), np.exp(-Th1)*csqrt(I1)
        q2 = np.sqrt(2*I2)*np.cos(Th2)
        p2 = -np.sqrt(2*I2)*np.sin(Th2)
        q3 = np.sqrt(2*I3)*np.cos(Th3)
        p3 = -np.sqrt(2*I3)*np.sin(Th3)
        return np.array([np.real(el) for el in [q1, q2, q3, p1, p2, p3]])


''' Computer Algebra Functions '''
# Using a dict representation of polynomials, where keys are tuples denoting the multi-index
# and values are the associated monomial coefficients.


def legendre_coeff(n):
    ''' Return the Legendre coefficients of Leg. Polynomial of degree n (up to 32), output is list
        [x^0_coeff, x^1_coeff, ..., x^n_coeff] - note that nan corresponds to 0 '''
    coeff = np.zeros(n+1)
    for k in range(n+1):
        coeff[k] = 2 ** n * binom(n, k) * binom(0.5 * (n + k - 1), n)
    return [fmpq(*c.as_integer_ratio()) for c in coeff]


def binom(n, k):
    ''' Binomial coefficient for kth term in binomial expansion of (a+b)^n'''
    prod = 1
    for i in range(1, k + 1):
        prod *= (1 / i) * (n + 1 - i)
    return prod


def Poisson_bracket(H, G):
    dHdZ = []
    dGdZ = []
    for j in range(6):
        dHdZ.append(H.deriv(j))
        dGdZ.append(G.deriv(j))
    summand = []
    for i in range(3):
        this_term = dHdZ[i]*dGdZ[3+i] - dHdZ[3+i]*dGdZ[i]
        summand.append(this_term)
    return summand[0] + summand[1] + summand[2]

''' Functions: Hamiltonian Expansion about L1 '''

def ham_expand(n):
    ''' Provides the Hamiltonian expansion term due to gravitation at order n in scaled/recentered coords. '''
    coeff = float2q(c_n(n)) * np.array(legendre_coeff(n))
    lctx = fmpq_mpoly_ctx.get(('x', 'rs')) # variables are x and rho^2
    lpoly = lctx.from_dict( dict(zip([(n-2*i, i) for i in range(n//2+1)], [coeff[-(2*i+1)] for i in range(n//2+1)])) )
    ctx = fmpq_mpoly_ctx.get(('j','q1','q2','q3','p1','p2','p3'))
    subx = ctx.from_dict( {(0,1,0,0,0,0,0): 1}) # Substitute: x = q1
    subrs= ctx.from_dict( {(0,2,0,0,0,0,0): 1,  # Substitute: rho^2 = x^2 + y^2 + z^2
                           (0,0,2,0,0,0,0): 1,
                           (0,0,0,2,0,0,0): 1})
    return Poly(lpoly.compose(subx, subrs))


def H_n(n, stage='recenter'):
    ''' Recentered Hamiltonian as an infinite series, truncated at order n, represented in
        polynomial form as a dict of monomial-coefficient pairs '''
    # Everything before the sum
    if stage == 'recenter':  # H = 1/2 (px^2 + py^2 + pz^2) + y*px - x*py - sum_{k>=2}(c_n*rho^n*L_n(x/rho))
        H0 = {(0, 0, 0, 2, 0, 0): 0.5,
              (0, 0, 0, 0, 2, 0): 0.5,
              (0, 0, 0, 0, 0, 2): 0.5,
              (0, 1, 0, 1, 0, 0): 1.0,
              (1, 0, 0, 0, 1, 0): -1.0}
        P0 = Poly(H0)
        # The sum, involving real coeffs, range to eq point (rho), and Legendre polynomials
        P = P0
        for k in range(2, n + 1):
            P += -ham_expand(k)
        return P
    elif stage == 'diagonalize':  # H = l*x*px + w1/2*(y^2 + py^2) + w2/2*(z^2 + pz^2) + sum_{k>=3}(Hk)
        D, lam, w1, w2 = diag_linear()
        H2 = {(1, 0, 0, 1, 0, 0): lam,
              (0, 2, 0, 0, 0, 0): w1 / 2,
              (0, 0, 0, 0, 2, 0): w1 / 2,
              (0, 0, 2, 0, 0, 0): w2 / 2,
              (0, 0, 0, 0, 0, 2): w2 / 2}
        P2 = Poly(H2)
        if n < 3: return P2
        P=P2
        M = D # Coordinate transformation matrix: Here, just diagonalization
        for k in range(3, n + 1):
            P += (-ham_expand(k)).transform(M)  # The term of 'recenter' Hamiltonian at order k
        return P
    elif stage == 'complexify':
        D, lam, w1, w2 = diag_linear()
        C = cmplx_xfrm()
        H2 = {(1, 0, 0, 1, 0, 0): lam,
              (0, 1, 0, 0, 1, 0): 1j * w1,
              (0, 0, 1, 0, 0, 1): 1j * w2}
        P2 = Poly(H2)
        if n < 3: return P2
        P=P2  # Will be list [H3, H4, H5, ..., Hn] under coord transformation x,y,z -> f(q1, q2, q3)
        M = D@C # Coordinate transformation matrix: Here, diagonalization and complexification
        for k in range(3, n + 1):
            P += (-ham_expand(k)).transform(M)
        return P


def generator(H, degree):
    ''' Determines the generator function G, a polynomial of degree ~degree~, which kills as many
        monomials of K of the same degree as is possible (i.e., non-resonant monomials).
        H should be provided already in 'by-degree' ordering. '''
    Hk = H[degree].poly
    _, lam, w1, w2 = diag_linear()
    nu = np.array([lam, 1j*w1, 1j*w2])
    Gk = dict(zip([m[1:] for m in Hk.monoms()], np.zeros(len(Hk.monoms()))))
    for mono in Hk.monoms():
        kq = np.array([int(m) for m in mono[1:4]]) # [kq1, kq2, kq3]
        kp = np.array([int(m) for m in mono[4:]])  # [kp1, kp2, kp3]
        if (degree % 2) == 0 and (kq == kp).all(): # Resonant term
            continue
        else: # Non-resonant term: Eliminate
            hk = q2float(Hk[mono]) * (1j)**int(mono[0])
            gk = -hk/np.dot((kp-kq), nu)
            Gk[mono[1:]] += gk
    return Poly(Gk)


def Lg(H, G, n=1):
    ''' Returns the n-fold nested Lie derivative of H with respect to G, multiplied by (1/(n!)) '''
    out = H
    for i in range(n):
        out = Poisson_bracket(out, G)
    return 1/fac(n) * out


def expLg(K, G, n, N=10):
    ''' Applies Lie series transformation of the generator G upon the polynomial K (where K is already ordered by degree).
        Provide n, the degree of generator G. Truncates the transformed polynomial for degrees above N. This gets faster as n -> N! '''
    out = dict(zip(K.keys(), K.values()))  # Temporary variable - don't mutate input
    for deg in K.keys():
        for i in range(1, 1+int((N-int(deg))/(n-2))): # i.e., solve deg+i*n-2*i=N for maximum val of i
            # Expected degree of the Poisson bracket output: deg + i*n - 2*i
            term_deg = deg + i*n - 2*i
            if term_deg > N or term_deg < n:  # Begin transform at degree of G; discard remainder terms of deg > N
                continue
            term = Lg(K[deg], G, i)
            out[term_deg] += term
    return out


def expLg_qp(Gs, N, reverse=False):
    ''' Applies Lie series transformation of the list of sequential generators, Gs, upon the canonical variables.
        The output is six polynomials encoding the transformation which can be evaluated at the initial coords to
        execute it. '''
    qp = [RealPoly({(1,0,0,0,0,0): 1.0}),
          RealPoly({(0,1,0,0,0,0): 1.0}),
          RealPoly({(0,0,1,0,0,0): 1.0}),
          RealPoly({(0,0,0,1,0,0): 1.0}),
          RealPoly({(0,0,0,0,1,0): 1.0}),
          RealPoly({(0,0,0,0,0,1): 1.0}) ]
    qp_out = qp
    G_list = [Gs[-i] for i in range(1, len(Gs)+1)] if reverse else Gs
    dirn = -1 if reverse else +1
    for k in range(6):
        print('Developing %s transformation in variable %i...'%('forward' if reverse else 'backward', k+1))
        for j,G in enumerate(G_list):
            n = [d for d in G.by_degree().keys()][0]
            qpk = dict(zip(qp_out[k].by_degree().keys(), qp_out[k].by_degree().values()))
            for deg in qpk.keys():
                for i in range(1,1+int((N-1)/(int(n)-2))):
                    term_deg = deg + i*n - 2*i
                    if term_deg > N: # or term_deg < :
                        continue
                    term = Lg(qpk[deg], dirn*G, i)
                    qp_out[k] += term
    return qp_out


def lieSeriesXfrm(H, N):
    ''' H is the complex-diagonal recentered Hamiltonian expansion up to order N, ordered by degree. Initial normalization order i=2.
        The desired normalization order is N. '''
    H_i_bd = H
    Gs = []
    Hs = []
    for i in range(3,N+1):
        print('Determining generator of degree %i...'%i)
        Gk = generator(H_i_bd, i)
        Gs.append(Gk)
        print('Applying Lie series transformation...')
        H_i_bd = expLg(H_i_bd, Gk, i, N)
        Hs.append(H_i_bd)
    return Gs, Hs


def canonical_flow(t, Z, dGdZ):
    ''' Equation of motion for the canonical flow over G defining the normal transformation. '''
    q1, q2, q3, p1, p2, p3 = Z
    J = np.block([[np.zeros((3,3)), np.eye(3)],
                  [-np.eye(3), np.zeros((3,3))]])
    dgdz = np.array([dGdZ[i].eval(Z, array=True) for i in range(6)])
    return J @ dgdz


def normal_xform(Z, data, reverse=False, verbose=True):
    ''' Integrates transformation EOM to perform the transformation from complex-diagonal
        coordinates centered on L1 to real normal coordinates. '''
    if type(data[0]) == list: # dGkdZ provided
        dGkdZ = data
        if reverse: # z_normal -> z_diag
            tau = +1
            deg_range = np.flip(np.arange(0, len(dGkdZ)))
        else: # z_diag -> z_normal
            tau = -1
            deg_range = np.arange(0, len(dGkdZ))
        if verbose:
            print('Flowing coordinates along transformation for fictitious time tau = %i...' % (tau))
        for i in deg_range:
            Z = solve_ivp(canonical_flow, [0, tau], Z, t_eval=[tau], args=(dGkdZ[i],)).y.T[-1]
        if verbose: print('Done.')
        return Z
    else:
        method = 'poly' # Pqp_* provided
        Pqp = data
        return np.array([np.real(Pqp[i].eval(Z, array=True)) for i in range(6)])


def cart2AA(X, data, reverse=False): # Seems to be something wrong
    if not reverse:
        Z_canon = cart2canon(X)
        Z_scale = scale_recenter(Z_canon, gamma=gamma_1)
        Z_diag = diagonalize(Z_scale)
        Z_norm = normal_xform(Z_diag, data=data)
        I = np.real(canon2AA(Z_norm)[:3])
        Th = [np.real(th)+(1-int(np.isclose(np.imag(th), 0)))*np.imag(th)*1j \
              for th in canon2AA(Z_norm)[3:]]
        return np.concatenate((I,Th))
    elif reverse: # Compose coordinate transformation in reverse order
        AA = X
        Z_norm = canon2AA(AA, reverse=True)
        Z_diag = normal_xform(Z_norm, data=data, reverse=True, verbose=False)
        Z_scale = diagonalize(Z_diag, reverse=True)
        Z_canon = scale_recenter(Z_scale, gamma=gamma_1, reverse=True)
        X_cart = cart2canon(Z_canon, reverse=True)
        return X_cart


''' Utility Functions '''

def CR3BP(t, state):
    mu=mu_EM
    x, y, z, xdot, ydot, zdot = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    xdd = 2 * ydot + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3
    ydd = -2 * xdot + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    zdd = -(1 - mu) * z / r1**3 - mu * z / r2**3

    return np.array([xdot, ydot, zdot, xdd, ydd, zdd])


def RTBprop(tspan, state,options={'rtol':1e-12, 'atol':1e-12}):
    return solve_ivp(CR3BP,(tspan[0],tspan[-1]),state,t_eval=tspan,**options).y


def save(item, fname, item_type='ham', overwrite=False):
    ''' Save sorted polynomial as .pkl. Input item should match output of by_degree(). '''
    if item_type in ['ham', 'hamiltonian']:
        if 'cache/hamiltonians/' not in fname:
            fname = 'cache/hamiltonians/'+fname
    elif item_type in ['gen', 'generator']:
        if 'cache/generators/' not in fname:
            fname = 'cache/generators/'+fname
    if '.pkl' not in fname:
        fname += '.pkl'
    if type(item) in (Poly, RealPoly):
        item = item.by_degree()
    deg_keys = [str(k) for k in item.keys()]
    if type(eval(deg_keys[0])) == int:
        to_save = {}
        for k_str in deg_keys:
            k = int(k_str)
            deg_dict = item[k].poly.to_dict()
            mono_keys = [str(mono) for mono in deg_dict.keys()]
            store = dict(zip(mono_keys, [(int(val.numer()), int(val.denom())) for val in deg_dict.values()]))
            to_save[k_str] = store
    elif type(eval(deg_keys[0])) == tuple:
        to_save = dict(zip(tuple([str(key) for key in item.keys()]), \
                           [(int(val.numer()), int(val.denom())) for val in item.values()]))
    try:
        with open(fname, 'rb') as f:
            pass
        if not overwrite:
            print('Filename already exists, specify overwrite if needed.')
            return
        else:
            with open(fname, 'wb') as f:
                # f.write(orjson.dumps(to_save))
                pkl.dump(to_save, f)
            print('Successfully overwrote %s to ' % item_type + fname)
            return
    except:
        with open(fname, 'wb') as f:
                # f.write(orjson.dumps(to_save))
                pkl.dump(to_save, f)
        print('Successfully wrote %s to '%item_type+fname)

def load(fname, item_type='ham'):
    ''' Load sorted polynomial from .pkl. Output item will match output of by_degree(). '''
    if item_type in ['ham', 'hamiltonian']:
        if 'cache/hamiltonians/' not in fname:
            fname = 'cache/hamiltonians/'+fname
    elif item_type in ['gen', 'generator']:
        if 'cache/generators/' not in fname:
            fname = 'cache/generators/'+fname
    if '.pkl' not in fname:
        fname += '.pkl'
    with open(fname, 'rb') as f:
        item = pkl.load(f)
    keys = [eval(key) for key in item.keys()]
    if type(keys[0]) == int:
        ctx = fmpq_mpoly_ctx.get(('j', 'q1', 'q2', 'q3', 'p1', 'p2', 'p3'))
        keys = [eval(key) for key in item.keys()]
        vals = []
        for k in keys:
            mono_keys = [eval(mono) for mono in item[str(k)].keys()]
            coeffs = [fmpq(*val) for val in item[str(k)].values()]
            vals.append( Poly(ctx.from_dict( dict(zip(mono_keys, coeffs)) )) )
        return dict(zip(keys, vals))
    elif type(keys[0]) == tuple:
        coeffs = [jnp.float64(val[0]) if val[1]==0 else jnp.complex128(val[0] + val[1]*1j) \
                      for val in item.values()]
        return dict(zip(keys, coeffs))

def save_grad(grad, N, overwrite=False):
    ''' Naming convention: dGkdZ_maxorder_thisorder_coord.pkl '''
    for i,grad_i in enumerate(grad):
        for j in range(6):
            save(grad_i[j], 'numerical/dGkdZ_%i_%i_%i.pkl'%(N, i+3, j), 'gen', overwrite=overwrite)
    return

def load_grad(N):
    ''' To Load: Identify your max order N. '''
    N = 12
    dGkdZ_ = []
    for i in range(3, N+1):
        grad_i = []
        for j in range(6):
            fname = 'dGkdZ_12_%i_%i.pkl'%(i, j)
            grad_i.append(Poly(load(fname, 'gen')))
        dGkdZ_.append(grad_i)
    return dGkdZ_

def save_Pqp(Pqp, dirn, N, overwrite=False):
    if dirn in [+1, 'fwd', 'forward']:
        dirn == 'fwd'
    elif dirn in [-1, 'bwd', 'backward', 'reverse']:
        dirn == 'bwd'
    for j,poly in enumerate(Pqp):
        save(Pqp[j], 'analytic/Pqp_%s_%i_%i'%(dirn, N, j), 'gen', overwrite=overwrite)
    return

def load_Pqp(dirn, N):
    if dirn in [+1, 'fwd', 'forward']:
        dirn == 'fwd'
    elif dirn in [-1, 'bwd', 'backward', 'reverse']:
        dirn == 'bwd'
    Pqp = []
    for j in range(6):
        Pqp.append(Poly(load('Pqp_%s_%i_%i'%(dirn, N, j), 'gen')))
    return Pqp