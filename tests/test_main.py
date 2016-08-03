import numpy as np
import numpy.testing as npt
import pytest

import ofit


class TestClass(object):

    def test_wrong_usage(self):
        with pytest.raises(ValueError):
            ofit.make_delay(location="ttop")
        with pytest.raises(ValueError):
            ofit.make_phase("x", location="tofp")

    def test_mult(selfself):
        cross = ofit.make_crosser()
        coupler = ofit.make_coupler()
        result = cross*coupler

        target = np.eye(4, dtype=complex)
        target_inner = np.sqrt(0.5) * np.array([[-1j, 1], [1, -1j]])
        target[1:3, 1:3] = target_inner

        npt.assert_array_almost_equal_nulp(result.matrix.astype(complex), target)



