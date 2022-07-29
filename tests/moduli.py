import numpy as np

from singdistance import SplittingInfo, ModuliSpace, logger

logger.setLevel('WARNING')


def test_zero_dimension():
    m1 = ModuliSpace(np.pi / 3, [2 * np.pi / 3])
    assert np.abs(m1.thurston_volume - 1) < 1e-5

    m1.calc_area_distribution()
    volume = ((m1.a_distribution.steps[1:] - m1.a_distribution.steps[:-1]) * m1.a_distribution.values).sum()
    assert np.abs(volume - m1.thurston_volume) < 1e-5


def test_set_splittings():
    m1 = ModuliSpace(2 * np.pi / 3, [np.pi / 3] * 5)
    m1.set_splittings()
    assert not m1.rhs_splittings[((0, 1), (2, 3, 4))].is_empty


def test_splitting_info():
    si = SplittingInfo(2 * np.pi / 3, [2 * np.pi / 3], [2 * np.pi / 3])
    assert not si.is_empty
    si.calc_rhs()

    val = si.rhs_function(.5)
    assert val > 0


def test_splitting_info_straightforward():
    si = SplittingInfo(2 * np.pi / 3, [2 * np.pi / 3], [2 * np.pi / 3])
    assert not si.is_empty
    si.calc_rhs(fast_one_dim=False)

    val = si.rhs_function(.5)
    assert val > 0


def test_one_dimension():
    m1 = ModuliSpace(np.pi / 3, [np.pi / 3] * 2)
    m1.ode_span = 0, 10
    m1.ode_steps = 10000
    m1.calc_area_distribution()

    volume = ((m1.a_distribution.steps[1:] - m1.a_distribution.steps[:-1]) * m1.a_distribution.values).sum()
    assert np.abs(volume - m1.thurston_volume) < 1e-2


def test_two_dimension():
    m1 = ModuliSpace(3, (2, 2, 2))
    m1.set_splittings()
    for si in m1.rhs_splittings.values():
        si.calc_rhs(beta_steps=lambda x: np.linspace(*x, 10))
    m1.calc_area_distribution()

    volume = ((m1.a_distribution.steps[1:] - m1.a_distribution.steps[:-1]) * m1.a_distribution.values).sum()
    assert np.abs(volume - m1.thurston_volume) < 1e-1


if __name__ == '__main__':
    test_zero_dimension()
    test_set_splittings()
    test_splitting_info()
    test_splitting_info_straightforward()
    test_one_dimension()
    test_two_dimension()
