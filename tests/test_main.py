import pytest
from ofit import ofit


class TestClass(object):

    def test_wrong_usage(self):
        with pytest.raises(ValueError):
            ofit.create_delay(location="ttop")
        with pytest.raises(ValueError):
            ofit.create_phase("x", location="tofp")
