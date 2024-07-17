from dataclasses import dataclass
from wos.fwd import *
from drjit import exp, log, sqrt, pi

'''
Green's function and Poisson kernel for 2D and 3D ball
'''

ACC = 40.
BIGNO = 1e10
BIGNI = 1e-10


def bessj0(x):
    '''
    Evaluate Bessel function of first kind and order 0 at input x
    '''
    ax = dr.abs(x)

    def _if(x):
        y = x * x
        ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
                                                          + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
                                                        + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
        ans = ans1 / ans2
        return ans

    def _else(x):
        z = 8.0 / ax
        y = z * z
        xx = ax - 0.785398164
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
                                                  + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
                                       + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                                                                      - y * 0.934935152e-7)))
        ans = sqrt(0.636619772 / ax) * (dr.cos(xx)
                                        * ans1 - z * dr.sin(xx) * ans2)
        return ans
    return dr.select(ax < 8.0, _if(x), _else(x))


def bessj1(x):
    '''
    Evaluate Bessel function of first kind and order 1 at input x
    '''
    ax = dr.abs(x)

    def _if(x):
        y = x * x
        ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
                                                              + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))))
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
                                                         + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        ans = ans1 / ans2
        return ans

    def _else(x):
        z = 8.0 / ax
        y = z * z
        xx = ax - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                                             + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3
                                    + y * (0.8449199096e-5 + y * (-0.88228987e-6
                                           + y * 0.105787412e-6)))
        ans = sqrt(0.636619772 / ax) * (dr.cos(xx)
                                        * ans1 - z * dr.sin(xx) * ans2)
        ans = dr.select(x < 0.0, -ans, ans)
        return ans
    return dr.select(ax < 8.0, _if(x), _else(x))


def bessj(n, x):
    '''
    Evaluate Bessel function of first kind and order n at input x
    '''
    ax = dr.abs(x)
    if n == 0:
        return bessj0(ax)
    if n == 1:
        return bessj1(ax)

    def _if():
        return 0.0

    def _elif():
        tox = 2.0 / ax
        bjm = bessj0(ax)
        bj = bessj1(ax)
        for j in range(1, n):
            bjp = j * tox * bj - bjm
            bjm = bj
            bj = bjp
        ans = bj
        return ans

    def _else():
        tox = 2.0 / ax
        m = 2 * ((n + int(sqrt(ACC * n))) // 2)
        # print(m)
        jsum = False
        bjp = ans = sum = 0.0
        bj = 1.0
        for j in range(m, 0, -1):
            bjm = j * tox * bj - bjp
            bjp = bj
            bj = bjm
            # print(j, bjm, bjp, bj)
            bjp = dr.select(dr.abs(bj) < BIGNO, bjp, bjp * BIGNI)
            ans = dr.select(dr.abs(bj) < BIGNO, ans, ans * BIGNI)
            sum = dr.select(dr.abs(bj) < BIGNO, sum, sum * BIGNI)
            bj = dr.select(dr.abs(bj) < BIGNO, bj, bj * BIGNI)
            # print(j, bj, bjp, ans, sum)
            if jsum:
                sum += bj
            jsum = not jsum
            if j == n:
                ans = bjp
        sum = 2.0 * sum - bj
        ans /= sum
        return ans
    ans = _else()
    ans = dr.select(ax == 0.0, _if(), ans)
    ans = dr.select(ax > n, _elif(), ans)
    return dr.select((x < 0.0) & (n % 2 == 1), -ans, ans)


def bessy0(x):
    '''
    Evaluate Bessel function of second kind and order 0 at input x
    '''
    def _if(x):
        y = x * x
        ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6
                                                        + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))))
        ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438
                                                       + y * (47447.26470 + y * (226.1030244 + y * 1.0))))
        ans = (ans1 / ans2) + 0.636619772 * bessj0(x) * log(x)
        return ans

    def _else(x):
        z = 8.0 / x
        y = z * z
        xx = x - 0.785398164
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
                                                  + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
                                       + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                                                                      + y * (-0.934945152e-7))))
        ans = sqrt(0.636619772 / x) * (dr.sin(xx)
                                       * ans1 + z * dr.cos(xx) * ans2)
        return ans
    return dr.select(x < 8.0, _if(x), _else(x))


def bessy1(x):
    '''
    Evaluate Bessel function of second kind and order 1 at input x
    '''
    def _if(x):
        y = x * x
        ans1 = x * (-0.4900604943e13 + y * (0.1275274390e13
                                            + y * (-0.5153438139e11 + y * (0.7349264551e9
                                                                           + y * (-0.4237922726e7 + y * 0.8511937935e4)))))
        ans2 = 0.2499580570e14 + y * (0.4244419664e12
                                      + y * (0.3733650367e10 + y * (0.2245904002e8
                                                                    + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))))
        ans = (ans1 / ans2) + 0.636619772 * (bessj1(x) * dr.log(x) - 1.0 / x)
        return ans

    def _else(x):
        z = 8.0 / x
        y = z * z
        xx = x - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                                             + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3
                                    + y * (0.8449199096e-5 + y * (-0.88228987e-6
                                           + y * 0.105787412e-6)))
        ans = sqrt(0.636619772 / x) * (dr.sin(xx)
                                       * ans1 + z * dr.cos(xx) * ans2)
        return ans
    return dr.select(x < 8.0, _if(x), _else(x))


def bessy(n, x):
    '''
    Evaluate Bessel function of second kind and order n at input x
    '''
    if n == 0:
        return bessy0(x)
    if n == 1:
        return bessy1(x)

    tox = 2.0 / x
    by = bessy1(x)
    bym = bessy0(x)
    for j in range(1, n):
        byp = j * tox * by - bym
        bym = by
        by = byp

    return by


def bessi0(x):
    '''
    Evaluate modified Bessel function of first kind and order 0 at input x
    '''
    ax = abs(x)
    # if ax < 3.75:

    def _if(x):
        y = x/3.75
        y = y*y
        ans = 1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
                                               + y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))))
        return ans
    # else:

    def _else(x):
        y = 3.75/ax
        ans = (exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
                                                + y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                                                                                    + y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                                                                                                                          + y*0.392377e-2))))))))
        return ans
    return dr.select(ax < 3.75, _if(x), _else(x))


def bessi1(x):
    '''
    Evaluate modified Bessel function of first kind and order 1 at input x
    '''
    ax = dr.abs(x)

    def _if(x):
        y = x / 3.75
        y = y * y
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                                                                   + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))))
        return ans

    def _else(x):
        y = 3.75 / ax
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
                                                       - y * 0.420059e-2))
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
                                                     + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
        ans *= (exp(ax) / sqrt(ax))
        return ans
    ans = dr.select(ax < 3.75, _if(x), _else(x))
    return dr.select(x < 0.0, -ans, ans)


def bessi(n, x):
    '''
    Evaluate modified Bessel function of first kind and order n at input x
    '''

    if n == 0:
        return bessi0(x)
    if n == 1:
        return bessi1(x)

    def _if():
        return 0.0

    def _else():
        tox = 2.0 / dr.abs(x)
        bip = ans = 0.0
        bi = 1.0
        for j in range(2 * (n + int(sqrt(ACC * n))), 0, -1):
            bim = bip + j * tox * bi
            bip = bi
            bi = bim
            ans = dr.select(dr.abs(bi) > BIGNO, ans, ans * BIGNI)
            bip = dr.select(dr.abs(bi) > BIGNO, bip, bip * BIGNI)
            bi = dr.select(dr.abs(bi) > BIGNO, bi, bi * BIGNI)
            if j == n:
                ans = bip
        ans *= bessi0(x) / bi
        return dr.select(x < 0.0 and n % 2 == 1, -ans, ans)
    return dr.select(dr.eq(x, 0.), _if(), _else())


def bessk0(x):
    '''
    Evaluate modified Bessel function of second kind and order 0 at input x
    '''
    # if x <= 2.0:
    def _if(x):
        y = x*x/4.0
        ans = (-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420
                                                      + y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
                                                                                         + y*(0.10750e-3+y*0.74e-5))))))
        return ans
    # else:

    def _else(x):
        y = 2.0/x
        ans = (exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
                                               + y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
                                                                                     + y*(-0.251540e-2+y*0.53208e-3))))))
        return ans
    return dr.select(x <= 2.0, _if(x), _else(x))


def bessk1(x):
    def _if(x):
        y = x * x / 4.0
        ans = (log(x / 2.0) * bessi1(x)) + (1.0 / x) * (1.0 + y * (0.15443144
                                                                   + y * (-0.67278579 + y * (-0.18156897 + y * (-0.1919402e-1
                                                                                                                + y * (-0.110404e-2 + y * -0.4686e-4))))))
        return ans

    def _else(x):
        y = 2.0 / x
        ans = (exp(-x) / sqrt(x)) * (1.25331414 + y * (0.23498619
                                                       + y * (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2
                                                                                                       + y * (0.325614e-2 + y * -0.68245e-3))))))
        return ans
    return dr.select(x <= 2.0, _if(x), _else(x))


def bessk(n, x):
    '''
    Evaluate modified Bessel function of second kind and order n at input x 
    '''
    if n == 0:
        return bessk0(x)
    if n == 1:
        return bessk1(x)

    tox = 2.0 / x
    bkm = bessk0(x)
    bk = bessk1(x)
    for j in range(1, n):
        bkp = bkm + j * tox * bk
        bkm = bk
        bk = bkp

    return bk


def G(sigma, R, r):
    muR = R * sqrt(sigma)
    K0muR = bessk0(muR)
    I0muR = bessi0(muR)
    mur = r * sqrt(sigma)
    K0mur = bessk0(mur)
    I0mur = bessi0(mur)
    return (K0mur - (I0mur / I0muR) * K0muR) / (2.0 * pi)


def P(sigma, R):
    muR = R * sqrt(sigma)
    I0muR = bessi0(muR)
    return 1.0 / (2.0 * R * pi * I0muR)


def G3D(sigma, R, r):
    muR = R * sqrt(sigma)
    expmuR = exp(-muR)
    sinhmuR = (1.0 - expmuR * expmuR) / (2.0 * expmuR)
    mur = r * sqrt(sigma)
    expmur = exp(-mur)
    sinhmur = (1.0 - expmur * expmur) / (2.0 * expmur)
    return (expmur - expmuR * sinhmur / sinhmuR) / (4.0 * pi * r)


def P3D(sigma, R):
    muR = R * sqrt(sigma)
    expmuR = exp(-muR)
    sinhmuR = (1.0 - expmuR * expmuR) / (2.0 * expmuR)
    return muR / (4.0 * pi * R*R * sinhmuR)


@dataclass
class GreensFnBall2D:
    R: float = 1.0
    rClamp: float = 1e-4


@dataclass
class HarmonicGreensFnBall2D(GreensFnBall2D):
    def G(self, x):
        r = dr.norm(x)
        return dr.log(self.R / dr.maximum(r, self.rClamp)) / (2.0 * pi)

    # def G_off_centered(self, c, x, y):
    #     r = dr.maximum(self.rClamp, dr.norm(x - y))
    #     return (dr.log(self.R * self.R - dr.dot(x - c, y - c)) - dr.log(self.R * r)) / (2.0 * pi)

    def G_off_centered(self, c, x, y):
        x = (x - c) / self.R
        y = (y - c) / self.R
        xy = x - y
        r = dr.norm(xy)
        x_star = x / dr.dot(x, x)
        return (dr.log(r) - dr.log(dr.norm(y - x_star)) - dr.log(dr.norm(x))) / (2.0 * pi)

    def G_off_centered2(self, c, x, y):
        x = (x - c) / self.R
        y = (y - c) / self.R
        theta = dr.atan2(x[1], x[0])
        theta0 = dr.atan2(y[1], y[0])
        r = dr.norm(x)
        r0 = dr.norm(y)
        cos = dr.cos(theta - theta0)
        return 1. / (4. * dr.pi) * \
            dr.log((r * r + r0 * r0 - 2. * r * r0 * cos) /
                   (r * r * r0 * r0 + 1. - 2. * r * r0 * cos))

    def P(self, x):
        return 1.0 / (2.0 * pi)

    def P_off_centered(self, c, x, y):
        x = (x - c) / self.R
        y = (y - c) / self.R
        thetax = dr.atan2(x[1], x[0])
        thetay = dr.atan2(y[1], y[0])
        r = dr.norm(x)
        ret = 1. / (2. * dr.pi) * (1 - r * r) / \
            (r * r + 1 - 2 * r * dr.cos(thetax - thetay))
        return ret / self.R

    def P_off_centered_der(self, c, x, y):
        x = (x - c) / self.R
        y = (y - c) / self.R
        thetax = dr.atan2(x[1], x[0])
        thetay = dr.atan2(y[1], y[0])
        r = dr.norm(x)
        cos = dr.cos(thetax - thetay)
        deno = r * r + 1 - 2 * r * cos
        ret = (r * r * cos - 2 * r + cos) / (dr.pi * deno * deno)
        return ret / self.R


# @dataclass
# class OffCenteredGreensBall:
#     sigma: float = 1.0
#     c: Array2 = Array2(0.)
#     R: float = 1.0
#     n: int = 100

#     def __post_init__(self):
#         self.muR = self.R * sqrt(self.sigma)

#     def G(self, x, y):
#         d1 = x - self.c
#         d2 = y - self.c
#         r1 = dr.norm(d1)
#         r2 = dr.norm(d2)
#         r = dr.norm(x - y)
#         r_minus = dr.minimum(r1, r2)
#         r_plus = dr.maximum(r1, r2)
#         mur_plus = r_plus * sqrt(self.sigma)
#         theta = dr.acos(dr.dot(d1, d2) / (r1 * r2))
#         res = Float(0.)
#         for i in range(self.n):
#             KmuR = bessk(i, self.muR)
#             Kmur_plus = bessk(i, mur_plus)
#             ImuR = bessi(i, self.muR)
#             Imur_plus = bessi(i, mur_plus)
#             res += dr.cos(i * theta) * ImuR(r - dr.sqrt(self.sigma)) * \
#                 (Kmur_plus - KmuR / ImuR * Imur_plus)
#         res /= 2.0 * pi
#         return res
