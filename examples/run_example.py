import itertools

import numpy as np

from src.distrs import Distribution2D, calc_distribution_function_1d, combinations, run_solve, run_convolve, logger

if __name__ == '__main__':
    alpha = [np.pi / 180 * 10, np.pi / 180 * 15, np.pi / 180 * 25, np.pi / 180 * 40]
    dphi = 1e-3
    darea = 1e-4

    logger.info('setting distribution sizes')

    distributions = dict()
    for i in range(1, len(alpha) + 1):
        distributions[i] = dict()
        for p in itertools.combinations(range(len(alpha)), i):
            area_max = np.tan(sum([alpha[i] for i in p]) / 4) / 2 + 2 * darea
            distributions[i][p] = Distribution2D([alpha[j] for j in p],
                                                 (0, sum([alpha[j] for j in p])), dphi,
                                                 (0, area_max), darea)

    logger.info('calculating distributions for length 1')

    for distr in distributions[1].values():
        logger.info(f'calculating distribution for {distr.alpha}')
        triangle_areas = 1 / (1 / np.tan(distr.phi1 / 2) + 1 / np.tan(distr.phi2 / 2))

        indexes = (triangle_areas // distr.darea).astype(int)
        indexes = np.minimum(indexes, len(distr.area) - 1)
        indexes = np.maximum(indexes, 0)

        distr.distr = np.zeros_like(distr.Area)
        for i in range(len(distr.distr)):
            index_max = indexes[max(i - 1, 0):min(i + 2, len(distr.distr) - 1)].max()
            index_min = indexes[max(i - 1, 0):min(i + 2, len(distr.distr) - 1)].min()
            if True or index_min < 50:
                distr.distr[i, indexes[i]] += 1 / distr.darea
            else:
                distr.distr[i, index_min:index_max + 1] = 1 / (index_max - index_min + 1) / distr.darea

    logger.info('calculating distributions for length 2')

    for distr in distributions[2].values():
        logger.info(f'calculating distribution for {distr.alpha}')
        for i, phi1 in enumerate(distr.phi1):
            dist_function = calc_distribution_function_1d(phi1, distr.alpha)
            for j, area in enumerate(distr.area[::]):
                distr.distr[i, j] = dist_function(area)
        distr.distr = np.maximum(distr.distr, 0)

    logger.info('calculating rhs for length 3')

    for p, distr in list(distributions[3].items()):
        logger.info(f'calculating rhs for {distr.alpha}')
        distr.rhs[:] = 0

        for comb in combinations(p):
            if len(comb) in {0, len(p)}:
                continue

            comb_rem = tuple(sorted(set(p) - set(comb)))
            distr1 = distributions[len(comb)][comb]
            distr2 = distributions[len(comb_rem)][comb_rem]

            logger.info(f'calculating rhs_summand for {distr.alpha}, combination {comb}')

            arg1 = distr1.distr * (1 / np.tan(distr1.Phi1 / 2) + 1 / np.tan(distr1.Phi2 / 2)) * distr1.Area ** len(
                distr1.alpha)
            arg2_ = distr2.distr * distr2.Area ** len(distr2.alpha)

            arg1 = np.nan_to_num(arg1)

            threads_x, threads_y = distr.distr.shape[0] // 30 + 1, distr.distr.shape[1] // 30 + 1

            rhs = 0
            rhs += run_convolve(np.zeros_like(distr.distr), arg2_, arg1,
                                kernel_params=((threads_x, threads_y), (30, 30)))
            rhs /= np.maximum(distr.Area, 1e-6) ** len(distr.alpha)
            rhs *= (distr2.dphi1 * distr2.darea)

            distr.rhs += rhs

    logger.info('calculating distributions for length 3')

    for distr in distributions[3].values():
        logger.info(f'calculating distribution for {distr.alpha}')
        distr.distr[:] = run_solve(distr.distr, distr.Phi1, distr.Phi2, distr.Area, distr.rhs, distr.dphi1, distr.darea,
                                   len(distr.alpha), kernel_params=(200, 30))
        distr.distr = np.maximum(distr.distr, 0)

    logger.info('calculating rhs for length 4')

    for p, distr in distributions[4].items():
        logger.info(f'calculating rhs for {distr.alpha}')
        distr.rhs[:] = 0

        for comb in combinations(p):
            if len(comb) in {0, len(p)}:
                continue

            comb_rem = tuple(sorted(set(p) - set(comb)))
            distr1 = distributions[len(comb)][comb]
            distr2 = distributions[len(comb_rem)][comb_rem]

            logger.info(f'calculating rhs_summand for {distr.alpha}, combination {comb}')

            arg1 = distr1.distr * (1 / np.tan(distr1.Phi1 / 2) + 1 / np.tan(distr1.Phi2 / 2)) * distr1.Area ** len(
                distr1.alpha)
            arg2_ = distr2.distr * distr2.Area ** len(distr2.alpha)

            arg1 = np.nan_to_num(arg1)

            threads_x, threads_y = distr.distr.shape[0] // 30 + 1, distr.distr.shape[1] // 30 + 1

            rhs = 0
            rhs += run_convolve(np.zeros_like(distr.distr), arg2_, arg1,
                                kernel_params=((threads_x, threads_y), (30, 30)))
            rhs /= np.maximum(distr.Area, 1e-6) ** len(distr.alpha)
            rhs *= (distr2.dphi1 * distr2.darea)

            distr.rhs += rhs

    logger.info('calculating distributions for length 4')

    for distr in distributions[4].values():
        logger.info(f'calculating distribution for {distr.alpha}')
        distr.distr[:] = run_solve(distr.distr, distr.Phi1, distr.Phi2, distr.Area, distr.rhs, distr.dphi1, distr.darea,
                                   len(distr.alpha), kernel_params=(1500, 4))
        distr.distr = np.maximum(distr.distr, 0)

    with open('result.npy', 'wb') as f:
        np.save(f, distributions[4][(0, 1, 2, 3)])
