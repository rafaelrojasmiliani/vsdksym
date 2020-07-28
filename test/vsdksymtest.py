""" Test very symple direct kinematics for symbolic computation
"""
import numpy as np
import sympy as sp
import unittest

import functools
import traceback
import sys
import pdb

from vsdksym.vsdksym import cVsdkSym


def debug_on(*exceptions):
    if not exceptions:
        exceptions = (Exception, )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)

    @debug_on()
    def testdk(self):
        ''' Test constructor and __call__'''

        dk = cVsdkSym()

        for i in range(7):
            a, d = np.random.rand(2)
            alpha, theta = (2.0 * np.random.rand(2) - 1.0) * np.pi
            dk.add_link(a, d, alpha, theta)

        dk()

    @debug_on()
    def testjac(self):
        ''' Test evaluation and correctednes of the jacobian'''
        print()
        dk = cVsdkSym()
        dim = 3
        for i in range(dim):
            a, d, alpha, theta = sp.symbols(
                'a_{:d} d_{:d} apha_{:d} theta_{:d}'.format(*(4 * [i])),
                real=True)
            dk.add_link(a, d, alpha, theta)

        jacsym = dk.jac()
        dksym = dk()

        q = dk.q_

        for i in range(3):
            for j in range(dim):
                testjac = dksym[i, 3].diff(q[j])
                print('testing equality {:d}, {:d}'.format(i, j))
                res = testjac-jacsym[i, j]
                res = sp.simplify(res)
                print(res)

                assert res == 0

    @debug_on()
    def testjac_new_vars(self):
        ''' Test evaluation and correctednes of the jacobian'''
        print()
        dk = cVsdkSym()
        dim = 3
        x = sp.symbols('x_0:{:d}'.format(dim), real=True)
        for i in range(dim):
            a, d, alpha, theta = sp.symbols(
                'a_{:d} d_{:d} apha_{:d} theta_{:d}'.format(*(4 * [i])),
                real=True)
            dk.add_link(a, d, alpha, theta, x[i])

        jacsym = dk.jac()
        dksym = dk()

        q = dk.q_

        for i in range(3):
            for j in range(dim):
                testjac = dksym[i, 3].diff(q[j])
                print('testing equality {:d}, {:d}'.format(i, j))
                res = testjac-jacsym[i, j]
                res = sp.simplify(res)
                print(res)

                assert res == 0


def main():
    unittest.main()


if __name__ == '__main__':
    main()
