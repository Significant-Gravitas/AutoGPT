from typing import Any

from AFAAS.interfaces.tools.context_items import ContextItem

ToolReturnValue = Any
ToolOutput = ToolReturnValue | tuple[ToolReturnValue, ContextItem]
