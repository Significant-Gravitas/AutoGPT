"""LLM catalog — catalog-as-code model management.

The canonical catalog file (``catalog.py``) holds model facts, per-model
credit costs, and routing cells; ``load_catalog()`` builds the in-process
view every consumer reads through.
"""

from .catalog import get_catalog
from .catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogCreator,
    CatalogModel,
    CatalogModelCost,
    CatalogPayload,
    CatalogProvider,
)
from .registry import (
    RegistryModel,
    RegistryModelMetadata,
    get_all_models,
    get_model,
    get_model_by_date_stripped_slug,
    get_route,
    has_models,
    is_loaded,
    load_catalog,
)

__all__ = [
    # Catalog
    "get_catalog",
    "CATALOG_SCHEMA_VERSION",
    "CatalogCreator",
    "CatalogModel",
    "CatalogModelCost",
    "CatalogPayload",
    "CatalogProvider",
    # Registry view
    "RegistryModel",
    "RegistryModelMetadata",
    "get_all_models",
    "get_model",
    "get_model_by_date_stripped_slug",
    "get_route",
    "has_models",
    "is_loaded",
    "load_catalog",
]
