from .graph import NodeModel
from .integrations import Webhook  # noqa: F401

# Resolve Webhook <- NodeModel forward reference
NodeModel.model_rebuild()
