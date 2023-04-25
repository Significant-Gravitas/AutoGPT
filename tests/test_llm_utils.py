import pytest
from openai.error import APIError, RateLimitError

from autogpt.llm_utils import retry_openai_api


@pytest.fixture(params=[RateLimitError, APIError])
def error(request):
    if request.param == APIError:
        return request.param("Error", http_status=502)
    else:
        return request.param("Error")


def error_factory(error_instance, error_count, retry_count, warn_user=True):
    class RaisesError:
        def __init__(self):
            self.count = 0

        @retry_openai_api(
            num_retries=retry_count, backoff_base=0.001, warn_user=warn_user
        )
        def __call__(self):
            self.count += 1
            if self.count <= error_count:
                raise error_instance
            return self.count

    return RaisesError()


def test_retry_open_api_no_error(capsys):
    @retry_openai_api()
    def f():
        return 1

    result = f()
    assert result == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_retry_open_api_passing(capsys, error):
    error_count = 2
    retry_count = 10

    raises = error_factory(error, error_count, retry_count)
    result = raises()
    call_count = min(error_count, retry_count) + 1
    assert result == call_count
    assert raises.count == call_count

    output = capsys.readouterr()

    if type(error) == RateLimitError:
        assert "Reached rate limit, passing..." in output.out
        assert "Please double check" in output.out


def test_retry_open_api_passing_edge(capsys, error):
    error_count = 2
    retry_count = 2

    raises = error_factory(error, error_count, retry_count)
    result = raises()
    call_count = min(error_count, retry_count) + 1
    assert result == call_count
    assert raises.count == call_count

    output = capsys.readouterr()

    if type(error) == RateLimitError:
        assert "Reached rate limit, passing..." in output.out
        assert "Please double check" in output.out


def test_retry_open_api_failing(capsys, error):
    error_count = 10
    retry_count = 2

    raises = error_factory(error, error_count, retry_count)

    with pytest.raises(type(error)):
        raises()

    call_count = min(error_count, retry_count) + 1
    assert raises.count == call_count

    output = capsys.readouterr()
    if type(error) == RateLimitError:
        assert "Reached rate limit, passing..." in output.out
        assert "Please double check" in output.out


def test_retry_open_api_failing_edge(capsys, error):
    error_count = 3
    retry_count = 2
    raises = error_factory(error, error_count, retry_count)

    with pytest.raises(type(error)):
        raises()

    call_count = min(error_count, retry_count) + 1
    assert raises.count == call_count

    output = capsys.readouterr()
    if type(error) == RateLimitError:
        assert "Reached rate limit, passing..." in output.out
        assert "Please double check" in output.out


def test_retry_open_api_failing_no_retries(capsys, error):
    error_count = 1
    retry_count = 0
    raises = error_factory(error, error_count, retry_count)

    with pytest.raises(type(error)):
        raises()

    call_count = min(error_count, retry_count) + 1
    assert raises.count == call_count

    output = capsys.readouterr()
    assert output.out == ""


def test_retry_open_api_rate_limit_no_warn(capsys):
    error_count = 2
    retry_count = 10

    raises = error_factory(RateLimitError, error_count, retry_count, warn_user=False)
    result = raises()
    call_count = min(error_count, retry_count) + 1
    assert result == call_count
    assert raises.count == call_count

    output = capsys.readouterr()

    assert "Reached rate limit, passing..." in output.out
    assert "Please double check" not in output.out


def test_retry_openapi_other_api_error(capsys):
    error_count = 2
    retry_count = 10

    raises = error_factory(APIError("Error", http_status=500), error_count, retry_count)

    with pytest.raises(APIError):
        raises()
    call_count = 1
    assert raises.count == call_count

    output = capsys.readouterr()
    assert output.out == ""
