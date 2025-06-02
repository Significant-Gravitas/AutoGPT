# Provider Enum Elimination Plan

## Executive Summary

The `ProviderName` enum in `backend/integrations/providers.py` currently requires manual updates when adding new providers, which conflicts with our SDK's goal of zero external configuration. This plan proposes a **hybrid solution** that maintains backward compatibility while enabling dynamic provider registration.

## Problem Analysis

### Current Issues
1. **Static Enum**: Python enums are immutable after class definition
2. **Manual Updates**: Each new provider requires updating the enum
3. **Type Safety**: FastAPI and Pydantic rely on the enum for validation
4. **Wide Usage**: Used in 50+ locations for type hints, validation, and database storage

### Why Dynamic Extension Fails
- Python enums cannot be modified at runtime
- Type checkers analyze enums statically
- FastAPI generates OpenAPI schemas at startup
- Circular imports prevent late modification

## Proposed Solution: Provider Registry Pattern

### Overview
Replace the static enum with a **provider registry** that:
1. Maintains core providers as constants
2. Allows runtime registration of new providers
3. Provides backward-compatible validation
4. Works seamlessly with FastAPI/Pydantic

### Architecture

```python
# backend/integrations/provider_registry.py
class ProviderRegistry:
    """Central registry for all providers"""
    
    # Core providers (backward compatibility)
    ANTHROPIC = "anthropic"
    GITHUB = "github"
    GOOGLE = "google"
    # ... all existing providers
    
    _registry: set[str] = {
        "anthropic", "github", "google", ...
    }
    
    @classmethod
    def register(cls, provider: str) -> None:
        """Register a new provider dynamically"""
        cls._registry.add(provider.lower())
    
    @classmethod
    def is_valid(cls, provider: str) -> bool:
        """Check if provider is registered"""
        return provider.lower() in cls._registry
    
    @classmethod
    def validate(cls, provider: str) -> str:
        """Validate and normalize provider name"""
        normalized = provider.lower()
        if not cls.is_valid(normalized):
            raise ValueError(f"Unknown provider: {provider}")
        return normalized
```

### Custom Pydantic Type

```python
# backend/integrations/provider_type.py
from typing import Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class Provider(str):
    """Custom Pydantic type for provider validation"""
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate(value: Any) -> str:
            if isinstance(value, ProviderName):  # Accept old enum
                return value.value
            if isinstance(value, str):
                return ProviderRegistry.validate(value)
            raise ValueError(f"Invalid provider type: {type(value)}")
        
        return core_schema.no_info_plain_validator_function(validate)
```

### Migration Strategy

1. **Phase 1**: Add registry alongside enum
2. **Phase 2**: Update type hints to use `Provider` type
3. **Phase 3**: Deprecate enum (keep for compatibility)
4. **Phase 4**: Remove enum in major version

## Implementation Checklist

### Phase 1: Core Implementation (Week 1)
- [ ] Create `backend/integrations/provider_registry.py` with ProviderRegistry class
- [ ] Create `backend/integrations/provider_type.py` with custom Pydantic type
- [ ] Add all existing enum values to registry initialization
- [ ] Create migration utilities to copy enum values to registry
- [ ] Add tests for provider registry functionality

### Phase 2: SDK Integration (Week 1)
- [ ] Update SDK decorators to use `ProviderRegistry.register()`
- [ ] Modify auto-registration to register providers via registry
- [ ] Update SDK imports to include Provider type
- [ ] Test that @provider decorator works with registry
- [ ] Verify backward compatibility with existing blocks

### Phase 3: Type Migration (Week 2)
- [ ] Create compatibility type alias: `ProviderNameType = Union[ProviderName, Provider]`
- [ ] Update critical paths to accept both enum and registry
- [ ] Start with webhook models: `provider: ProviderNameType`
- [ ] Update OAuth handlers to use provider type
- [ ] Update credential models to use provider type

### Phase 4: FastAPI Integration (Week 2)
- [ ] Create custom FastAPI query parameter validator
- [ ] Update route handlers to use Provider type
- [ ] Test OpenAPI schema generation
- [ ] Ensure proper error messages for invalid providers
- [ ] Update API documentation

### Phase 5: Database Layer (Week 3)
- [ ] Ensure database continues storing providers as strings
- [ ] Update Prisma models if needed
- [ ] Test database queries with new providers
- [ ] Verify migration compatibility
- [ ] Add database constraints if necessary

### Phase 6: Testing & Documentation (Week 3)
- [ ] Create comprehensive test suite for provider registry
- [ ] Test migration from enum to registry
- [ ] Performance test registry lookups
- [ ] Update developer documentation
- [ ] Create migration guide for existing code

### Phase 7: Gradual Rollout (Week 4)
- [ ] Deploy with feature flag for registry usage
- [ ] Monitor for any validation errors
- [ ] Gradually migrate internal code to use registry
- [ ] Collect feedback from block developers
- [ ] Plan enum deprecation timeline

## Code Examples

### Before (Current)
```python
# Must manually add to enum
class ProviderName(str, Enum):
    MYSERVICE = "myservice"  # Manual addition required

# Block usage
@provider("myservice")  # Fails - not in enum
class MyServiceBlock(Block):
    pass
```

### After (With Registry)
```python
# Automatic registration via decorator
@provider("myservice")  # Auto-registers in registry
class MyServiceBlock(Block):
    pass

# Or explicit registration
ProviderRegistry.register("myservice")
```

### Backward Compatible Usage
```python
# Old code continues to work
def process(provider: ProviderName):  # Still accepts enum
    pass

# New code uses registry
def process_new(provider: Provider):  # Accepts any registered provider
    pass

# Transition code
def process_both(provider: Union[ProviderName, Provider]):  # Accepts both
    pass
```

## Benefits

1. **Zero Configuration**: New providers auto-register via decorators
2. **Backward Compatible**: Existing code continues to work
3. **Type Safe**: Maintains validation and type checking
4. **Scalable**: No manual enum updates needed
5. **Flexible**: Supports dynamic provider discovery

## Risks and Mitigations

### Risk 1: Type Safety Loss
**Mitigation**: Custom Pydantic type maintains runtime validation

### Risk 2: Breaking Changes
**Mitigation**: Phased migration with compatibility layer

### Risk 3: Performance Impact
**Mitigation**: Set lookups are O(1), negligible performance impact

### Risk 4: OpenAPI Schema
**Mitigation**: Custom schema generation for dynamic providers

## Success Metrics

1. **Zero Manual Updates**: New providers require no code changes outside blocks
2. **100% Backward Compatibility**: All existing code continues to work
3. **Type Safety Maintained**: No increase in runtime errors
4. **Developer Satisfaction**: Easier provider addition process
5. **Performance**: No measurable performance degradation

## Conclusion

This plan provides a path to eliminate the static ProviderName enum requirement while maintaining all benefits of type safety and validation. The provider registry pattern aligns perfectly with the SDK's goal of zero external configuration and will enable true plug-and-play block development.