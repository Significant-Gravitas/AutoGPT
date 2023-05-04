from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="CheckWeatherUsingGETResponse200CurrentCondition")


@attr.s(auto_attribs=True)
class CheckWeatherUsingGETResponse200CurrentCondition:
    """
    Attributes:
        text (Union[Unset, str]): Weather condition text
        icon (Union[Unset, str]): Weather icon url
        code (Union[Unset, int]): Weather condition unique code
    """

    text: Union[Unset, str] = UNSET
    icon: Union[Unset, str] = UNSET
    code: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        text = self.text
        icon = self.icon
        code = self.code

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if text is not UNSET:
            field_dict["text"] = text
        if icon is not UNSET:
            field_dict["icon"] = icon
        if code is not UNSET:
            field_dict["code"] = code

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        text = d.pop("text", UNSET)

        icon = d.pop("icon", UNSET)

        code = d.pop("code", UNSET)

        check_weather_using_get_response_200_current_condition = cls(
            text=text,
            icon=icon,
            code=code,
        )

        check_weather_using_get_response_200_current_condition.additional_properties = d
        return check_weather_using_get_response_200_current_condition

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
