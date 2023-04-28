import pytest
from openai.error import APIError, RateLimitError

from autogpt.llm import COSTS, get_ada_embedding
from autogpt.llm.llm_utils import retry_openai_api


@pytest.fixture(params=[RateLimitError, APIError])
def error(request):
    if request.param == APIError:
        return request.param("Error", http_status=502)
    else:
        return request.param("Error")


@pytest.fixture
def mock_create_embedding(mocker):
    mock_response = mocker.MagicMock()
    mock_response.usage.prompt_tokens = 5
    mock_response.__getitem__.side_effect = lambda key: [{"embedding": [0.1, 0.2, 0.3]}]
    return mocker.patch(
        "autogpt.llm.llm_utils.create_embedding", return_value=mock_response
    )


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


@pytest.mark.parametrize(
    "error_count, retry_count, failure",
    [(2, 10, False), (2, 2, False), (10, 2, True), (3, 2, True), (1, 0, True)],
    ids=["passing", "passing_edge", "failing", "failing_edge", "failing_no_retries"],
)
def test_retry_open_api_passing(capsys, error, error_count, retry_count, failure):
    call_count = min(error_count, retry_count) + 1

    raises = error_factory(error, error_count, retry_count)
    if failure:
        with pytest.raises(type(error)):
            raises()
    else:
        result = raises()
        assert result == call_count

    assert raises.count == call_count

    output = capsys.readouterr()

    if error_count and retry_count:
        if type(error) == RateLimitError:
            assert "Reached rate limit, passing..." in output.out
            assert "Please double check" in output.out
        if type(error) == APIError:
            assert "API Bad gateway" in output.out
    else:
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


def test_get_ada_embedding(mock_create_embedding, api_manager):
    model = "text-embedding-ada-002"
    embedding = get_ada_embedding("test")
    mock_create_embedding.assert_called_once_with(
        "test", model="text-embedding-ada-002"
    )

    assert embedding == [0.1, 0.2, 0.3]

    cost = COSTS[model]["prompt"]
    assert api_manager.get_total_prompt_tokens() == 5
    assert api_manager.get_total_completion_tokens() == 0
    assert api_manager.get_total_cost() == (5 * cost) / 1000
