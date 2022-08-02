from singdistance import *


def test_analytic_1d():
    m1 = ModuliSpace(2, (2, 3))
    m1.ode_steps = 10000
    m1.calc_area_distribution()

    alpha1, alpha2 = [m1.alpha[0]], [m1.alpha[1]]

    m = np.exp(-1j * alpha1[0] / 2) / np.sin(alpha1[0] / 2) + \
        np.exp(1j * (alpha2[0] / 2 - m1.phi1)) / np.sin(alpha2[0] / 2)
    m = np.abs(m) / 2
    c = - 1 / np.tan(alpha1[0] / 2) / 2 - 1 / np.tan(alpha2[0] / 2) / 2

    def f_an(a):
        # https://tinyurl.com/ansolution
        q = m1.q

        w1 = ((1 - c * q) ** 2 - m ** 2 * q ** 2) ** .5
        w2 = (a - c) * (1 - c * q) - m ** 2 * q
        w3 = (m ** 2 - (a - c) ** 2) ** .5

        res = np.arctan(w2 / w3 / w1) / w1 / q / a ** 2
        res += np.arctan((c - a) / w3) / q / a ** 2
        return res

    dist = m1.a_distribution
    x = dist.steps[:-1]
    y = dist.values

    k1 = y[2000] / x[2000] ** -2  # choose point in the 4th part

    y_an = np.array([f_an(x) + k1 * x ** -2 for x in x])
    y_an = np.minimum(np.maximum(y_an, -5), 5)

    k2 = (y[200] - f_an(x[200]) / 2) / x[200] ** -2  # choose point in second part

    y_an2 = np.array([f_an(x) / 2 + k2 * x ** -2 for x in x])
    y_an2 = np.minimum(np.maximum(y_an2, -5), 5)

    y_an3 = k1 * x ** -2
    y_an3 = np.minimum(np.maximum(y_an3, -5), 5)

    y_an4 = 0 * x

    defect = (~np.logical_or.reduce([np.abs(y - z) < 1e-3 for z in [y_an, y_an2, y_an3, y_an4]], axis=0)).sum()
    assert defect / len(x) < 1e-3


if __name__ == '__main__':
    test_analytic_1d()
