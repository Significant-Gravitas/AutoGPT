import numpy as np
from numpy.testing import assert_array_equal, assert_array_compare

from numpy.random import SeedSequence


def test_reference_data():
    """ Check that SeedSequence generates data the same as the C++ reference.

    https://gist.github.com/imneme/540829265469e673d045
    """
    inputs = [
        [3735928559, 195939070, 229505742, 305419896],
        [3668361503, 4165561550, 1661411377, 3634257570],
        [164546577, 4166754639, 1765190214, 1303880213],
        [446610472, 3941463886, 522937693, 1882353782],
        [1864922766, 1719732118, 3882010307, 1776744564],
        [4141682960, 3310988675, 553637289, 902896340],
        [1134851934, 2352871630, 3699409824, 2648159817],
        [1240956131, 3107113773, 1283198141, 1924506131],
        [2669565031, 579818610, 3042504477, 2774880435],
        [2766103236, 2883057919, 4029656435, 862374500],
    ]
    outputs = [
        [3914649087, 576849849, 3593928901, 2229911004],
        [2240804226, 3691353228, 1365957195, 2654016646],
        [3562296087, 3191708229, 1147942216, 3726991905],
        [1403443605, 3591372999, 1291086759, 441919183],
        [1086200464, 2191331643, 560336446, 3658716651],
        [3249937430, 2346751812, 847844327, 2996632307],
        [2584285912, 4034195531, 3523502488, 169742686],
        [959045797, 3875435559, 1886309314, 359682705],
        [3978441347, 432478529, 3223635119, 138903045],
        [296367413, 4262059219, 13109864, 3283683422],
    ]
    outputs64 = [
        [2477551240072187391, 9577394838764454085],
        [15854241394484835714, 11398914698975566411],
        [13708282465491374871, 16007308345579681096],
        [15424829579845884309, 1898028439751125927],
        [9411697742461147792, 15714068361935982142],
        [10079222287618677782, 12870437757549876199],
        [17326737873898640088, 729039288628699544],
        [16644868984619524261, 1544825456798124994],
        [1857481142255628931, 596584038813451439],
        [18305404959516669237, 14103312907920476776],
    ]
    for seed, expected, expected64 in zip(inputs, outputs, outputs64):
        expected = np.array(expected, dtype=np.uint32)
        ss = SeedSequence(seed)
        state = ss.generate_state(len(expected))
        assert_array_equal(state, expected)
        state64 = ss.generate_state(len(expected64), dtype=np.uint64)
        assert_array_equal(state64, expected64)


def test_zero_padding():
    """ Ensure that the implicit zero-padding does not cause problems.
    """
    # Ensure that large integers are inserted in little-endian fashion to avoid
    # trailing 0s.
    ss0 = SeedSequence(42)
    ss1 = SeedSequence(42 << 32)
    assert_array_compare(
        np.not_equal,
        ss0.generate_state(4),
        ss1.generate_state(4))

    # Ensure backwards compatibility with the original 0.17 release for small
    # integers and no spawn key.
    expected42 = np.array([3444837047, 2669555309, 2046530742, 3581440988],
                          dtype=np.uint32)
    assert_array_equal(SeedSequence(42).generate_state(4), expected42)

    # Regression test for gh-16539 to ensure that the implicit 0s don't
    # conflict with spawn keys.
    assert_array_compare(
        np.not_equal,
        SeedSequence(42, spawn_key=(0,)).generate_state(4),
        expected42)
