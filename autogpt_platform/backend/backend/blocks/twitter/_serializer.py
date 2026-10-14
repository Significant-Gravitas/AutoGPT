from typing import Any


class BaseSerializer:
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Helper method to serialize individual values"""
        if hasattr(value, "data"):
            return value.data
        return value


class IncludesSerializer(BaseSerializer):
    @classmethod
    def serialize(cls, includes: dict[str, Any]) -> dict[str, Any]:
        """Serializes the includes dictionary"""
        if not includes:
            return {}

        serialized_includes = {}
        for key, value in includes.items():
            if isinstance(value, list):
                serialized_includes[key] = [
                    cls._serialize_value(item) for item in value
                ]
            else:
                serialized_includes[key] = cls._serialize_value(value)

        return serialized_includes


class ResponseDataSerializer(BaseSerializer):
    @classmethod
    def serialize_dict(cls, item: dict[str, Any]) -> dict[str, Any]:
        """Serializes a single dictionary item"""
        serialized_item = {}

        if hasattr(item, "__dict__"):
            items = item.__dict__.items()
        else:
            items = item.items()

        for key, value in items:
            if isinstance(value, list):
                serialized_item[key] = [
                    cls._serialize_value(sub_item) for sub_item in value
                ]
            else:
                serialized_item[key] = cls._serialize_value(value)

        return serialized_item

    @classmethod
    def serialize_list(cls, data: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Serializes a list of dictionary items"""
        if not data:
            return []
        return [cls.serialize_dict(item) for item in data]


class ResponseSerializer:
    @classmethod
    def serialize(cls, response) -> dict[str, Any]:
        """Main serializer that handles both data and includes"""
        result = {"data": None, "included": {}}

        # Handle response.data
        if response.data:
            if isinstance(response.data, list):
                result["data"] = ResponseDataSerializer.serialize_list(response.data)
            else:
                result["data"] = ResponseDataSerializer.serialize_dict(response.data)

        # Handle includes
        if hasattr(response, "includes") and response.includes:
            result["included"] = IncludesSerializer.serialize(response.includes)

        return result
