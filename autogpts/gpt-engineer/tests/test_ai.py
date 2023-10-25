import pytest

from gpt_engineer.core.ai import AI


@pytest.mark.xfail(reason="Constructor assumes API access")
def test_ai():
    AI()
    # TODO Assert that methods behave and not only constructor.
