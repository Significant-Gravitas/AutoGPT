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
    cast,
)

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
  from ..models.check_weather_using_get_response_200_current import (
      CheckWeatherUsingGETResponse200Current,
  )
  from ..models.check_weather_using_get_response_200_location import (
      CheckWeatherUsingGETResponse200Location,
  )





T = TypeVar("T", bound="CheckWeatherUsingGETResponse200")


@attr.s(auto_attribs=True)
class CheckWeatherUsingGETResponse200:
    """ 
        Attributes:
            location (Union[Unset, CheckWeatherUsingGETResponse200Location]):
            current (Union[Unset, CheckWeatherUsingGETResponse200Current]):
            info_link (Union[Unset, str]): A link to a page with more information about the location's weather in the format
                https://weathergpt.vercel.app/{location}.
     """

    location: Union[Unset, 'CheckWeatherUsingGETResponse200Location'] = UNSET
    current: Union[Unset, 'CheckWeatherUsingGETResponse200Current'] = UNSET
    info_link: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        from ..models.check_weather_using_get_response_200_current import (
            CheckWeatherUsingGETResponse200Current,
        )
        from ..models.check_weather_using_get_response_200_location import (
            CheckWeatherUsingGETResponse200Location,
        )
        location: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.location, Unset):
            location = self.location.to_dict()

        current: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.current, Unset):
            current = self.current.to_dict()

        info_link = self.info_link

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if location is not UNSET:
            field_dict["location"] = location
        if current is not UNSET:
            field_dict["current"] = current
        if info_link is not UNSET:
            field_dict["infoLink"] = info_link

        return field_dict



    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.check_weather_using_get_response_200_current import (
            CheckWeatherUsingGETResponse200Current,
        )
        from ..models.check_weather_using_get_response_200_location import (
            CheckWeatherUsingGETResponse200Location,
        )
        d = src_dict.copy()
        _location = d.pop("location", UNSET)
        location: Union[Unset, CheckWeatherUsingGETResponse200Location]
        if isinstance(_location,  Unset):
            location = UNSET
        else:
            location = CheckWeatherUsingGETResponse200Location.from_dict(_location)




        _current = d.pop("current", UNSET)
        current: Union[Unset, CheckWeatherUsingGETResponse200Current]
        if isinstance(_current,  Unset):
            current = UNSET
        else:
            current = CheckWeatherUsingGETResponse200Current.from_dict(_current)




        info_link = d.pop("infoLink", UNSET)

        check_weather_using_get_response_200 = cls(
            location=location,
            current=current,
            info_link=info_link,
        )

        check_weather_using_get_response_200.additional_properties = d
        return check_weather_using_get_response_200

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
