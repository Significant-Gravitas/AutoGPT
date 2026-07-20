"""LLM catalog — catalog-as-code model management.

The canonical catalog file (``catalog.py``) holds model facts, per-model
credit costs, and routing cells; ``load_catalog()`` builds the in-process
view every consumer reads through.
"""

from .catalog import CATALOG
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
    RegistryModelCreator,
    RegistryModelMetadata,
    get_all_model_slugs_for_validation,
    get_all_models,
    get_default_model_slug,
    get_enabled_models,
    get_model,
    get_route,
    get_schema_options,
    load_catalog,
)

__all__ = [
    # Catalog
    "CATALOG",
    "CATALOG_SCHEMA_VERSION",
    "CatalogCreator",
    "CatalogModel",
    "CatalogModelCost",
    "CatalogPayload",
    "CatalogProvider",
    # Registry view
    "RegistryModel",
    "RegistryModelCreator",
    "RegistryModelMetadata",
    "get_all_model_slugs_for_validation",
    "get_all_models",
    "get_default_model_slug",
    "get_enabled_models",
    "get_model",
    "get_route",
    "get_schema_options",
    "load_catalog",
]
