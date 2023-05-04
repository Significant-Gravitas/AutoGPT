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

T = TypeVar("T", bound="CheckWeatherUsingGETResponse200Location")


@attr.s(auto_attribs=True)
class CheckWeatherUsingGETResponse200Location:
    """ 
        Attributes:
            name (Union[Unset, str]):
            region (Union[Unset, str]):
            country (Union[Unset, str]):
            lat (Union[Unset, float]):
            lon (Union[Unset, float]):
            tz_id (Union[Unset, str]):
            localtime_epoch (Union[Unset, int]):
            localtime (Union[Unset, str]):
     """

    name: Union[Unset, str] = UNSET
    region: Union[Unset, str] = UNSET
    country: Union[Unset, str] = UNSET
    lat: Union[Unset, float] = UNSET
    lon: Union[Unset, float] = UNSET
    tz_id: Union[Unset, str] = UNSET
    localtime_epoch: Union[Unset, int] = UNSET
    localtime: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        region = self.region
        country = self.country
        lat = self.lat
        lon = self.lon
        tz_id = self.tz_id
        localtime_epoch = self.localtime_epoch
        localtime = self.localtime

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if name is not UNSET:
            field_dict["name"] = name
        if region is not UNSET:
            field_dict["region"] = region
        if country is not UNSET:
            field_dict["country"] = country
        if lat is not UNSET:
            field_dict["lat"] = lat
        if lon is not UNSET:
            field_dict["lon"] = lon
        if tz_id is not UNSET:
            field_dict["tz_id"] = tz_id
        if localtime_epoch is not UNSET:
            field_dict["localtime_epoch"] = localtime_epoch
        if localtime is not UNSET:
            field_dict["localtime"] = localtime

        return field_dict



    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        region = d.pop("region", UNSET)

        country = d.pop("country", UNSET)

        lat = d.pop("lat", UNSET)

        lon = d.pop("lon", UNSET)

        tz_id = d.pop("tz_id", UNSET)

        localtime_epoch = d.pop("localtime_epoch", UNSET)

        localtime = d.pop("localtime", UNSET)

        check_weather_using_get_response_200_location = cls(
            name=name,
            region=region,
            country=country,
            lat=lat,
            lon=lon,
            tz_id=tz_id,
            localtime_epoch=localtime_epoch,
            localtime=localtime,
        )

        check_weather_using_get_response_200_location.additional_properties = d
        return check_weather_using_get_response_200_location

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
