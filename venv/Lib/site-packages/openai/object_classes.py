from openai import api_resources
from openai.api_resources.experimental.completion_config import CompletionConfig

OBJECT_CLASSES = {
    "engine": api_resources.Engine,
    "experimental.completion_config": CompletionConfig,
    "file": api_resources.File,
    "fine-tune": api_resources.FineTune,
    "model": api_resources.Model,
    "deployment": api_resources.Deployment,
}
