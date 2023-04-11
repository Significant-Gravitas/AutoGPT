
from numpy.distutils.from_template import process_str
from numpy.testing import assert_equal


pyf_src = """
python module foo
    <_rd=real,double precision>
    interface
        subroutine <s,d>foosub(tol)
            <_rd>, intent(in,out) :: tol
        end subroutine <s,d>foosub
    end interface
end python module foo
"""

expected_pyf = """
python module foo
    interface
        subroutine sfoosub(tol)
            real, intent(in,out) :: tol
        end subroutine sfoosub
        subroutine dfoosub(tol)
            double precision, intent(in,out) :: tol
        end subroutine dfoosub
    end interface
end python module foo
"""


def normalize_whitespace(s):
    """
    Remove leading and trailing whitespace, and convert internal
    stretches of whitespace to a single space.
    """
    return ' '.join(s.split())


def test_from_template():
    """Regression test for gh-10712."""
    pyf = process_str(pyf_src)
    normalized_pyf = normalize_whitespace(pyf)
    normalized_expected_pyf = normalize_whitespace(expected_pyf)
    assert_equal(normalized_pyf, normalized_expected_pyf)
