import sage.all as sg

from singdistance import *


def calc_rhs_1d(self, fast_one_dim=True):
    if self.is_empty:
        return 0

    logger.debug(f'Calculating rhs for SI({self.phi1}, {self.phi2}, {self.alpha1}, {self.alpha2}) '
                 f'at {hex(id(self))}')

    if len(self.alpha1) == 1 and len(self.alpha2) == 1 and fast_one_dim:
        # fast computation for the 1-dimensional case
        logger.debug(f'SI at {hex(id(self))}: fast mode')

    def area(_beta):
        _beta_ = self.phi1 - _beta
        _beta2 = self.alpha1[0] - _beta
        _beta2_ = self.phi2 - _beta2
        a1 = 1 / (1 / (np.tan(_beta / 2 + 1e-6)) + 1 / np.tan(_beta2 / 2 + 1e-6))
        a2 = 1 / (1 / np.tan(_beta_ / 2 + 1e-6) + 1 / np.tan(_beta2_ / 2 + 1e-6))
        return a1 + a2

    if self.limits[1] > self.limits[0]:
        area_max, beta_max = sg.find_local_maximum(area(sg.var('x')), self.limits[0], self.limits[1])
        area0, area1 = area(self.limits[0]), area(self.limits[1])
    else:
        area0, area1, area_max, beta_max = 0, 0, 0, 0

    def area_diff(_beta):  # sg.derivative(area(sg.var('b')), sg.var('b'))
        _beta_ = self.phi1 - _beta
        _beta2 = self.alpha1[0] - _beta
        _beta2_ = self.phi2 - _beta2

        # noinspection DuplicatedCode
        return \
            1 / 2 * (
                    (np.tan(_beta / 2 + 1e-6) ** -2 + 1) -
                    (np.tan(_beta2 / 2 + 1e-6) ** -2 + 1)
            ) / (1 / np.tan(_beta / 2 + 1e-6) + 1 / np.tan(_beta2 / 2 + 1e-6)) ** 2 - \
            1 / 2 * (
                    (np.tan(_beta_ / 2 + 1e-6) ** -2 + 1) -
                    (np.tan(_beta2_ / 2 + 1e-6) ** -2 + 1)
            ) / (1 / np.tan(_beta_ / 2 + 1e-6) + 1 / np.tan(_beta2_ / 2 + 1e-6)) ** 2

    def rhs(s):
        val = 0
        if area0 < s < area_max:
            b = sg.find_root(area(sg.var('x')) - s, self.limits[0], beta_max)
            val += 1 / s / area_diff(b)
        if area1 < s < area_max:
            b = sg.find_root(area(sg.var('x')) - s, beta_max, self.limits[1])
            val -= 1 / s / area_diff(b)
        return val

    self.rhs_function = rhs


def test_rhs_1d():
    si = SplittingInfo(2, [3], [4])

    si.calc_rhs()
    rhs_basic = si.rhs_function

    calc_rhs_1d(si)
    rhs_new = si.rhs_function

    x = np.linspace(0, 1, 1000)
    rb = np.array([rhs_basic(x) for x in x])
    rn = np.array([rhs_new(x) for x in x])

    assert (np.abs(rn - rb) / (rb + 1e-6) > 1e-3).sum() < 10


if __name__ == '__main__':
    test_rhs_1d()
