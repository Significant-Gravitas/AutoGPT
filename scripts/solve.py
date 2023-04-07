from config import Config, Singleton
import wolframalpha

cfg = Config()

class WolframAlphaSolver(metaclass=Singleton):

  _client : wolframalpha.Client

  def __init__(self):
    wolframalpha_appid = cfg.wolframalpha_appid
    self._client = wolframalpha.Client(wolframalpha_appid)

  #
  # 
  #
  def query(self, queryString):
    """
    Query the Wolfram Alpha API with the query string
    """
    result = self._client.query(queryString)
    appendedResult = ""
    if result.success == False:
      # TODO handle "did you means"
      appendedResult += "No results found for this query"
    else:
      for pod in result.pods:
        if pod.title == "Result":
          appendedResult += pod.text
    return appendedResult
