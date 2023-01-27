import itertools
import logging

import numpy as np
from numba import cuda, void, float32, int32
from numba.cuda import cudamath
from numpy.fft import fft2, ifft2

LOG_LEVEL = logging.INFO
log_formatter = logging.Formatter('[%(asctime)s l%(lineno)s - %(funcName)10s] %(message)s')
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)
logger.propagate = False


@cuda.jit(void(float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32))
def calc_rhs_1dim(r, p1, p2, ar, alpha1):
    N, T = r.shape

    thread = cuda.threadIdx.x
    block = cuda.blockIdx.x
    n_threads = cuda.blockDim.x
    # n_blocks = cuda.gridDim.x

    for t in range(1, T):
        if block * n_threads + thread < N:
            phi1 = p1[block * n_threads + thread, t]
            phi2 = p2[block * n_threads + thread, t]
            alpha2 = phi1 + phi2 - alpha1
            a = ar[block * n_threads + thread, t]

            beta_min, beta_max = max(0, phi1 - alpha2), min(alpha1, phi1)

            m_x = cudamath.math.cos(alpha1 / 2) / cudamath.math.sin(alpha1 / 2) + cudamath.math.cos(
                alpha2 / 2 - phi1) / cudamath.math.sin(alpha2 / 2)
            m_y = -cudamath.math.sin(alpha1 / 2) / cudamath.math.sin(alpha1 / 2) + cudamath.math.sin(
                alpha2 / 2 - phi1) / cudamath.math.sin(alpha2 / 2)

            gamma = cudamath.math.atan2(m_y, m_x)
            m = (m_x ** 2 + m_y ** 2) ** .5 / 2

            q = 1 / cudamath.math.tan(alpha1 / 2)
            q += 1 / cudamath.math.tan(alpha2 / 2)
            q /= 2

            res = 0
            if abs((a + q) / m) < 1:
                value = 1 / a / m / (1 - ((a + q) / m) ** 2) ** .5
                am1, am2 = cudamath.math.cos(beta_min + gamma) * m - q, cudamath.math.cos(beta_max + gamma) * m - q
                n = 0
                if a > am1:
                    n += 1
                if a > am2:
                    n += 1
                res += n * value
            r[block * n_threads + thread, t] = res


@cuda.jit(void(float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32, float32, int32))
def solve(r, p1, p2, a, y, dp, da, n):
    N, T = r.shape

    thread = cuda.threadIdx.x
    block = cuda.blockIdx.x
    n_threads = cuda.blockDim.x
    # n_blocks = cuda.gridDim.x

    i = block * n_threads + thread

    if i < N:
        q = 1 / cuda.cudamath.math.tan(p1[i, 0] / 2)
        q += 1 / cuda.cudamath.math.tan(p2[i, 0] / 2)

    for t in range(1, T):
        if i < N:
            r[i, t] = r[i, t - 1]
            r[i, t] += da * ((2 * q - n / a[i, t]) * r[i, t - 1] +
                             y[i, t] / 2) / (1 - q * a[i, t])
            if q > 0 and a[i, t] > 1 / q:  # 1/q is maximal area when phi1+phi2 < 2pi
                r[i, t] = 0


@cuda.jit(void(float32[:, :], float32[:, :], float32[:, :]))
def convolve(r, arg1, arg2):
    N1, M1 = arg1.shape
    N2, M2 = arg2.shape

    # todo make splitting more equal
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    for i1 in range(N1):
        for k1 in range(M1):
            i2 = i - i1
            k2 = k - k1
            if 0 <= i2 < N2 and 0 <= k2 < M2:
                r[i, k] += arg1[i1, k1] * arg2[i2, k2]


def calc_distribution_function_1d(phi1, alpha):
    alpha1, alpha2 = tuple(alpha)
    phi2 = alpha1 + alpha2 - phi1

    limits = max(0, phi1 - alpha2), min(alpha1, phi1)

    def area(_beta):
        _beta_ = phi1 - _beta
        _beta2 = alpha1 - _beta
        _beta2_ = phi2 - _beta2
        a1 = 1 / (1 / (np.tan(_beta / 2)) + 1 / np.tan(_beta2 / 2))
        a2 = 1 / (1 / np.tan(_beta_ / 2) + 1 / np.tan(_beta2_ / 2))
        return a1 + a2

    if limits[0] < limits[1]:
        area0, area1 = area(limits[0]), area(limits[1])
    else:
        area0, area1 = 0., 0.

    m = np.exp(-1j * alpha1 / 2) / np.sin(alpha1 / 2) + \
        np.exp(1j * (alpha2 / 2 - phi1)) / np.sin(alpha2 / 2)
    m = np.abs(m) / 2
    c = - 1 / np.tan(alpha1 / 2) / 2 - 1 / np.tan(alpha2 / 2) / 2
    q = 1 / np.tan(phi1 / 2) + 1 / np.tan(phi2 / 2)

    def f_an(a):
        # https://tinyurl.com/ansolution

        w1 = ((1 - c * q) ** 2 - m ** 2 * q ** 2) ** .5
        w2 = (a - c) * (1 - c * q) - m ** 2 * q
        w3 = (m ** 2 - (a - c) ** 2) ** .5

        res = np.arctan(w2 / w3 / w1) / w1 / q / a ** 2
        res += np.arctan((c - a) / w3) / q / a ** 2
        return res

    k0 = -f_an(area0) / 2 * area0 ** 2
    k1 = -f_an(area1) / 2 * area1 ** 2

    def f_res(a):
        if limits[0] > limits[1]:
            return 0
        res = 0
        if m + c > a > area0:
            res += f_an(a) / 2
        if m + c > a > area1:
            res += f_an(a) / 2
        if q < 0 or a < 1 / q:
            if a > area0:
                res += k0 * a ** -2
            if a > area1:
                res += k1 * a ** -2
        return res

    return f_res


def run_rhs_1dim(res_, Phi1_, Phi2_, Area_, alpha1_, kernel_params=(1000, 1)):
    d_res = cuda.to_device(res_)
    d_Phi1_ = cuda.to_device(Phi1_)
    d_Phi2_ = cuda.to_device(Phi2_)
    d_Area_ = cuda.to_device(Area_)

    calc_rhs_1dim[kernel_params[0], kernel_params[1]](d_res, d_Phi1_, d_Phi2_, d_Area_, alpha1_)
    res_ = d_res.copy_to_host()
    return res_


def run_solve(res_, Phi1_, Phi2_, Area_, y_, dphi_, darea_, n_, kernel_params=(1000, 1)):
    # todo: Phi1, Phi2 -> phi1, phi2 (1-dimensional)
    if res_.shape[0] > kernel_params[0] * kernel_params[1]:
        raise RuntimeWarning('Not enough kernels')
    if res_.shape[0] * 2 + 1 < kernel_params[0] * kernel_params[1]:
        logger.warning('Too many kernels')
    d_res = cuda.to_device(res_)
    y_res = cuda.to_device(y_)
    d_Phi1_ = cuda.to_device(Phi1_)
    d_Phi2_ = cuda.to_device(Phi2_)
    d_Area_ = cuda.to_device(Area_)

    solve[kernel_params[0], kernel_params[1]](d_res, d_Phi1_, d_Phi2_, d_Area_, y_res, dphi_, darea_, n_)
    res__ = d_res.copy_to_host()
    return res__


def run_convolve(res_, arg1_, arg2_, kernel_params=((100, 100), (20, 20))):
    if res_.shape[0] > kernel_params[0][0] * kernel_params[1][0] or \
            res_.shape[1] > kernel_params[0][1] * kernel_params[1][1]:
        raise RuntimeWarning('Not enough kernels')

    d_res = cuda.to_device(res_)
    d_arg1_ = cuda.to_device(arg1_)
    d_arg2_ = cuda.to_device(arg2_)

    convolve[kernel_params[0], kernel_params[1]](d_res, d_arg1_, d_arg2_)
    res__ = d_res.copy_to_host()
    return res__


class Distribution2D:
    def __init__(self, alpha, phi1_range, phi1_step, area_range, area_step, dtype=np.float32):
        self.alpha = list(alpha)
        self.phi1_range = phi1_range
        self.phi1_step = phi1_step
        self.area_range = area_range
        self.area_step = area_step

        self.phi1 = np.arange(*phi1_range, phi1_step, dtype=dtype)
        self.phi2 = sum(alpha) - self.phi1
        self.area = np.arange(*area_range, area_step, dtype=dtype)

        self.dphi1, self.darea = self.phi1[1] - self.phi1[0], self.area[1] - self.area[0]
        self.Phi1, self.Area = np.meshgrid(self.phi1, self.area, indexing='ij')
        self.Phi2 = sum(alpha) - self.Phi1

        self.Q = 1 / np.tan(self.Phi1 / 2) + 1 / np.tan(self.Phi2 / 2)
        self.rhs = np.zeros([len(self.phi1), len(self.area)], dtype=dtype)
        self.distr = np.zeros([len(self.phi1), len(self.area)], dtype=dtype)


def fft_convolve(arg1, arg2):
    arg1_conv = np.zeros([arg1.shape[0] + arg2.shape[0] - 1 + 123, arg1.shape[1] + arg2.shape[1] - 1 + 123],
                         dtype=np.float32)
    arg1_conv[:arg1.shape[0], :arg1.shape[1]] = arg1

    return np.real(ifft2(fft2(arg1_conv) * fft2(arg2, s=arg1_conv.shape)))


def combinations(numbers):
    for i in range(len(numbers)):
        for comb in itertools.combinations(numbers, i):
            yield comb


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
            res *= np.maximum(0, 1 - sum(b) / 2 / np.pi) ** (len(b) - 1)
        vol += res
    vol *= (-4 * np.pi) ** (len(angles) - 3) / np.math.factorial(len(angles) - 2)
    return vol


def thurston_volume(angles):  # normalize McMullen's volume by 4 ^ dim(M)
    return mcmullen_volume(angles) / 4 ** (len(angles) - 3)
