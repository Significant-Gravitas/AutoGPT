import pinecone
from pinecone.config import Config


def _test_multi_init():
    env = 'test-env'
    api_key = 'foobar'
    # first init() sets api_key
    pinecone.init(api_key=api_key)
    assert Config.ENVIRONMENT == 'us-west1-gcp'
    # next init() shouldn't clobber api_key
    pinecone.init(environment=env)
    assert Config.ENVIRONMENT == env
    assert Config.API_KEY == api_key
