import sympy as sp
from sympy import cos, sin
from sympy.core.rules import Transform
from sympy import Float


class cVsdkSym(object):
    ''' Direct Kinematic symbolic function generator.  Very symple class meant
    to generate a symbolic sympy expression of the direct kinematic function of
    a robot and its jacobian.
    '''

    def __init__(self):
        self.mij_ = []
        self.m0j_ = []
        self.q_ = []
        self.dim_ = 0
        self.mee = sp.eye(4)

    def add_link(self, _a, _d, _alpha, _theta, _q=None):
        ''' Adds a new link to the robot with given DH
        parameters'''
        self.mij_.append(cDHmatrixSym(_a, _d, _alpha, _theta))
        self.m0j_.append(sp.zeros(4, 4))
        if _q is None:
            qs = sp.symbols('q[{:d}]'.format(len(self.q_)), real=True)
        else:
            qs = _q
        self.q_.append(qs)
        self.dim_ += 1

    def set_tcp_offset(self, _x, _y, _z):
        self.mee[0:3, 3] = _x, _y, _z

    def __call__(self, _q=None):

        if _q is None:
            q = self.q_
        else:
            q = _q

        self.m0j_[0] = self.mij_[0](q[0])

        for i, mij in enumerate(self.mij_[1:], start=1):
            self.m0j_[i] = self.m0j_[i - 1] * mij(q[i])

        res = self.m0j_[-1] * self.mee
        for i in range(3):
            for j in range(4):
                res[i, j] = res[i, j].xreplace(
                    Transform(lambda x: x.round(8),
                              lambda x: isinstance(x, Float)))
        return res

    def __len__(self):
        return len(self.mij_)

    def jac(self, _q=None):

        if _q is None:
            q = self.q_
        else:
            q = _q

        pe = self(q)[:3, -1]
        jac = sp.zeros(6, self.dim_)
        axis = sp.Matrix([0.0, 0.0, 1.0])
        p = sp.Matrix([0.0, 0.0, 0.0])
        for j in range(0, self.dim_):
            jac[:3, j] = axis.cross(pe - p).doit()
            jac[3:, j] = axis
            axis = self.m0j_[j][:3, 2]
            p = self.m0j_[j][:3, -1]

        for i in range(6):
            for j in range(self.dim_):
                jac[i, j] = jac[i, j].xreplace(
                    Transform(lambda x: x.round(8),
                              lambda x: isinstance(x, Float)))

        return jac


class cDHmatrixSym(object):
    ''' Symbolic DH transformation matrix generator '''

    def __init__(self, _a, _d, _alpha, _theta):
        self.buff_ = sp.zeros(4, 4)
        self.theta_ = _theta
        self.alpha_ = _alpha
        self.a_ = _a
        self.d_ = _d
        self.cosal_ = cos(_alpha)
        self.sinal_ = sin(_alpha)

    def __call__(self, _q):
        cth = cos(self.theta_ + _q)
        sth = sin(self.theta_ + _q)
        res = sp.zeros(4, 4)
        res[0, 0] = cth
        res[0, 1] = -sth * self.cosal_
        res[0, 2] = sth * self.sinal_
        res[0, 3] = self.a_ * cth

        res[1, 0] = sth
        res[1, 1] = cth * self.cosal_
        res[1, 2] = -cth * self.sinal_
        res[1, 3] = self.a_ * sth

        res[2, 1] = self.sinal_
        res[2, 2] = self.cosal_
        res[2, 3] = self.d_

        res[3, 3] = 1

        for i in range(3):
            for j in range(4):
                res[i, j] = res[i, j].xreplace(
                    Transform(lambda x: x.round(8),
                              lambda x: isinstance(x, Float)))
        return res


def dh_sym_matrix(_q, _a, _d, _alpha, _theta):
    theta_ = _theta
    alpha_ = _alpha
    a_ = _a
    d_ = _d
    cosal_ = cos(alpha_)
    sinal_ = sin(alpha_)

    cth = cos(theta_ + _q)
    sth = sin(theta_ + _q)
    res = sp.zeros(4, 4)
    res[0, 0] = cth
    res[0, 1] = -sth * cosal_
    res[0, 2] = sth * sinal_
    res[0, 3] = a_ * cth

    res[1, 0] = sth
    res[1, 1] = cth * cosal_
    res[1, 2] = -cth * sinal_
    res[1, 3] = a_ * sth

    res[2, 1] = sinal_
    res[2, 2] = cosal_
    res[2, 3] = d_

    res[3, 3] = 1
    return res
