import logging
from dataclasses import dataclass

import numpy as np
from sage.calculus.ode import ode_solver

LOG_LEVEL = logging.INFO
log_formatter = logging.Formatter('[%(asctime)s l%(lineno)s - %(funcName)10s] %(message)s')
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)


def partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller


def mcmullen_volume(angles):
    vol = 0
    for p in partitions(angles):
        if len(p) < 3:
            continue
        res = -(-1) ** len(p) * np.math.factorial(len(p) - 3)
        for b in p:
            res *= max(0, 1 - sum(b) / 2 / np.pi) ** (len(b) - 1)
        vol += res
    vol *= (-4 * np.pi) ** (len(angles) - 3) / np.math.factorial(len(angles) - 2)
    return vol


def thurston_volume(angles):  # normalize McMullen's volume by 4 ^ dim(M)
    return mcmullen_volume(angles) / 4 ** (len(angles) - 3)


def splittings(collection, ordered=True):
    if len(collection) == 0:
        yield [], []
        return

    for part1, part2 in splittings(collection[1:]):
        yield [collection[0]] + part1, part2
        if ordered:
            yield part1, [collection[0]] + part2


def rhs_1d(s, c, m, gamma, limits):
    if abs((s - c) / m) < 1:
        res = 1 / max(s, 1e-6) / m / (1 - ((s - c) / m) ** 2) ** .5
        num = 0
        beta1 = np.arccos((s - c) / m) - gamma
        beta2 = -np.arccos((s - c) / m) - gamma
        if limits[0] < beta1 < limits[1]:
            num += 1
        if limits[0] < beta2 < limits[1]:
            num += 1

        return res * num
    return 0


def density(beta, phi1, phi2, alpha1, alpha2, arg):
    beta_ = phi1 - beta
    beta2 = sum(alpha1) - beta
    beta2_ = phi2 - beta2

    m1 = ModuliSpace.get_from_cache(beta, alpha1, create=True)[1]
    m2 = ModuliSpace.get_from_cache(beta_, alpha2, create=True)[1]

    if not len(m1.a_distribution.steps):
        m1.calc_area_distribution()
    if not len(m2.a_distribution.steps):
        m2.calc_area_distribution()

    d1 = m1.a_distribution.rescale(arg).to_pdf()
    d2 = m2.a_distribution.rescale(arg).to_pdf()

    f1 = d1.values * d1.steps[:-1] ** (1 + m1.dimension)
    f2 = d2.values * d2.steps[:-1] ** (1 + m2.dimension)

    f = np.convolve(f1, f2) * (d1.steps[1] - d1.steps[0])
    f = f[:len(arg)]

    f *= sum([1 / np.tan(x / 2) for x in [beta, beta_, beta2, beta2_]])
    f /= arg ** (2 + m1.dimension + m2.dimension) + 1e-6

    return f


def calc_rhs(betas, phi1, phi2, alpha1, alpha2, arg):
    res = np.zeros_like(arg)

    for beta in betas:
        res += density(beta, phi1, phi2, alpha1, alpha2, arg)

    return res * (betas[1] - betas[0])


@dataclass
class Distribution:
    steps: np.array = np.array([])
    values: np.array = np.array([])
    mode: str = 'pdf'

    def to_cdf(self):
        if self.mode == 'cdf':
            return self
        elif self.mode == 'pdf':
            mode = 'cdf'
            values = np.cumsum(self.values * (self.steps[1:] - self.steps[:-1]))
            return Distribution(self.steps, values, mode)
        raise RuntimeError('Unknown mode!')

    def to_pdf(self):
        if self.mode == 'pdf':
            return self
        elif self.mode == 'cdf':
            mode = 'pdf'
            values = (self.values - np.insert(self.values[:-1], 0, 0)) / (self.steps[1:] - self.steps[:-1])
            return Distribution(self.steps, values, mode)
        raise RuntimeError('Unknown mode!')

    def rescale(self, steps):
        dist = self.to_cdf()
        dist.values = np.interp(steps, dist.steps, np.insert(dist.values, 0, 0))[1:]
        dist.steps = steps
        return dist

    def __radd__(self, other):
        if other == 0:
            return self
        if len(self.steps) != len(other.steps) or \
                np.abs(self.steps - other.steps).mean() > 1e-6 or \
                self.mode != other.mode:
            raise NotImplementedError('Cannot add distributions!')
        return Distribution(self.steps, self.values + other.values, self.mode)

    def __add__(self, other):
        return self.__radd__(other)


class SplittingInfo:
    def __init__(self, phi1, alpha1, alpha2):
        self.phi1 = phi1
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha1_sum = sum(alpha1)
        self.alpha2_sum = sum(alpha2)
        self.phi2 = self.alpha1_sum + self.alpha2_sum - phi1

        self.is_empty = len(alpha1) * len(alpha2) == 0

        self.limits = max(0, phi1 - self.alpha2_sum) + 1e-5, min(self.alpha1_sum, phi1) - 1e-5
        if self.limits[0] >= self.limits[1]:
            self.is_empty = True

        self.a_limits = (0, 2)
        self.a_steps = 1000

        self.rhs_function = None

    def calc_rhs(self, beta_steps=lambda x: np.linspace(*x, 1000), fast_one_dim=True):
        if self.is_empty:
            return 0

        logger.debug(f'Calculating rhs for SI({self.phi1}, {self.phi2}, {self.alpha1}, {self.alpha2}) '
                     f'at {hex(id(self))}')

        if len(self.alpha1) == 1 and len(self.alpha2) == 1 and fast_one_dim:
            # fast computation for the 1-dimensional case
            logger.debug(f'SI at {hex(id(self))}: fast mode')

            limits = max(0, self.phi1 - self.alpha2[0]) + 1e-5, min(self.alpha1[0], self.phi1) - 1e-5

            m = np.exp(-1j * self.alpha1[0] / 2) / np.sin(self.alpha1[0] / 2) + \
                np.exp(1j * (self.alpha2[0] / 2 - self.phi1)) / np.sin(self.alpha2[0] / 2)
            gamma = np.angle(m)
            m = np.abs(m) / 2
            c = - 1 / np.tan(self.alpha1[0] / 2) / 2 - 1 / np.tan(self.alpha2[0] / 2) / 2

            self.rhs_function = lambda s: rhs_1d(s, c, m, gamma, limits)

        else:
            # straightforward computation
            betas = beta_steps(self.limits)
            arg = np.linspace(*self.a_limits, self.a_steps)

            res = calc_rhs(betas, self.phi1, self.phi2, self.alpha1, self.alpha2, arg)

            rhs_dist = Distribution(arg, res, 'pdf')

            def rhs(s):
                i = np.argmin(np.abs(rhs_dist.steps[:-1] - s))
                return rhs_dist.values[i]

            self.rhs_function = rhs


class ModuliSpace:
    cache = []

    def __init__(self, phi1, alpha, ode_span=(0, 5), ode_steps=1000):
        self.alpha = alpha
        self.alpha_sum = sum(alpha)
        self.phi1 = phi1
        self.phi2 = self.alpha_sum - phi1
        logger.debug(f'ModuliSpace({self.phi1}, {self.phi2}, {self.alpha}) at {hex(id(self))}')

        self.thurston_volume = thurston_volume([2 * np.pi - phi1, 2 * np.pi - self.phi2, *alpha])
        self.dimension = len(alpha) - 1

        self.a_distribution = Distribution()
        self.l_distribution = Distribution()

        self.q = 1 / np.tan(self.phi1 / 2) + 1 / np.tan(self.phi2 / 2)

        self.rhs_splittings = dict()
        self.rhs = Distribution()
        self.is_empty = not all(x > 0 for x in list(alpha) + [phi1, self.phi2])

        self.ode_span = ode_span
        self.ode_steps = ode_steps

        self.rhs_function = None

        self.cache.append(self)

    def calc_area_distribution(self):
        if self.is_empty:
            raise RuntimeError('Moduli space is empty!')
        if self.dimension == 0:
            q = self.q
            if q > 0:
                self.a_distribution = \
                    Distribution(np.array([1 / q - 1e-6, 1 / q, 1 / q + 1e-6]), np.array([0, 1]), 'cdf')
                self.a_distribution = self.a_distribution.to_pdf()
        else:
            if not len(self.rhs_splittings):
                self.set_splittings()
                self.rhs = Distribution()
            else:
                logger.info(f'Splittings already set for ModuliSpace at {hex(id(self))}')

            for splitting in self.rhs_splittings:
                if not self.rhs_splittings[splitting].rhs_function:
                    self.rhs_splittings[splitting].calc_rhs()
                    self.rhs = Distribution()
                else:
                    logger.info(f'Rhs already calculated for SI at {hex(id(self.rhs_splittings[splitting]))}')

            if not self.rhs_function:
                def rhs(s):
                    return sum([si.rhs_function(s) for si in self.rhs_splittings.values() if not si.is_empty])

                self.rhs_function = rhs
            else:
                logger.info(f'ModuliSpace at {hex(id(self))} already has rhs_function')

            solver = ode_solver()
            solver.function = lambda s, f: [((2 * self.q - (self.dimension + 1) / max(s, 1e-6)) * f[0] +
                                             self.rhs_function(s) / 2) / (-s * self.q + 1)]
            solver.ode_solve(y_0=[0], t_span=self.ode_span, num_points=self.ode_steps)
            sol = np.array([[x, y[0]] for x, y in solver.solution])

            if self.q > 0:
                sol[sol[:, 0] > 1 / self.q, 1] = 0  # 1/q is maximal area when phi1+phi2 < 2pi

            self.a_distribution = Distribution(sol[:, 0], sol[:-1, 1])

            vol = ((self.a_distribution.steps[1:] - self.a_distribution.steps[:-1]) * self.a_distribution.values).sum()

            logger.debug(f'ModuliSpace at {hex(id(self))}: '
                         f'distribution volume {vol}; McMullen volume {self.thurston_volume}')

    def calc_length_distribution(self):
        if not len(self.a_distribution.steps):
            raise RuntimeError('First calculate distribution of area function!')
        steps = self.a_distribution.steps ** -.5
        values = 2 * self.a_distribution.values * self.a_distribution.steps[1:] ** (3 / 2)
        steps = steps[::-1]
        values = values[::-1]
        self.l_distribution = Distribution(steps, values)

    def set_splittings(self):
        for i1, i2 in splittings(range(len(self.alpha)), ordered=False):
            alpha1 = [self.alpha[index] for index in i1]
            alpha2 = [self.alpha[index] for index in i2]
            self.rhs_splittings[(tuple(i1), tuple(i2))] = SplittingInfo(self.phi1, alpha1, alpha2)

    @classmethod
    def get_from_cache(cls, phi1, alpha, precision=1e-3, cache=None, create=False):
        if not cache:
            cache = cls.cache
        for m in cache:
            if len(alpha) == len(m.alpha) and \
                    np.abs(np.array(alpha) - m.alpha).sum() < precision and \
                    (np.abs(phi1 - m.phi1) < precision or
                     np.abs(phi1 - m.phi2) < precision):
                logger.info(f'ModuliSpace for {phi1, alpha} found in cache: {hex(id(m))}')
                return True, m
        if create:
            logger.info(f'ModuliSpace for {phi1, alpha} not found')
            return False, ModuliSpace(phi1, alpha)
        return False, None
