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
  from ..models.check_weather_using_get_response_200_current_condition import (
      CheckWeatherUsingGETResponse200CurrentCondition,
  )





T = TypeVar("T", bound="CheckWeatherUsingGETResponse200Current")


@attr.s(auto_attribs=True)
class CheckWeatherUsingGETResponse200Current:
    """ 
        Attributes:
            last_updated (Union[Unset, str]): Local time when the real time data was updated
            last_updated_epoch (Union[Unset, int]): Local time when the real time data was updated in unix time
            temp_c (Union[Unset, float]): Temperature in celsius
            temp_f (Union[Unset, float]): Temperature in fahrenheit
            is_day (Union[Unset, int]): 1 = Yes 0 = No, Whether to show day condition icon or night icon
            condition (Union[Unset, CheckWeatherUsingGETResponse200CurrentCondition]):
            wind_mph (Union[Unset, float]): Wind speed in miles per hour
            wind_kph (Union[Unset, float]): Wind speed in kilometer per hour
            wind_degree (Union[Unset, int]): Wind direction in degrees
            wind_dir (Union[Unset, str]): Wind direction as 16 point compass, e.g., NSW
            pressure_mb (Union[Unset, float]): Pressure in millibars
            pressure_in (Union[Unset, float]): Pressure in inches
            precip_mm (Union[Unset, float]): Precipitation amount in millimeters
            precip_in (Union[Unset, float]): Precipitation amount in inches
            humidity (Union[Unset, int]): Humidity as percentage
            cloud (Union[Unset, int]): Cloud cover as percentage
            feelslike_c (Union[Unset, float]): Feels like temperature in celsius
            feelslike_f (Union[Unset, float]): Feels like temperature in fahrenheit
            vis_km (Union[Unset, float]): Visibility in kilometers
            vis_miles (Union[Unset, float]): Visibility in miles
            uv (Union[Unset, float]): UV Index
            gust_mph (Union[Unset, float]): Wind gust in miles per hour
            gust_kph (Union[Unset, float]): Wind gust in kilometer per hour
     """

    last_updated: Union[Unset, str] = UNSET
    last_updated_epoch: Union[Unset, int] = UNSET
    temp_c: Union[Unset, float] = UNSET
    temp_f: Union[Unset, float] = UNSET
    is_day: Union[Unset, int] = UNSET
    condition: Union[Unset, 'CheckWeatherUsingGETResponse200CurrentCondition'] = UNSET
    wind_mph: Union[Unset, float] = UNSET
    wind_kph: Union[Unset, float] = UNSET
    wind_degree: Union[Unset, int] = UNSET
    wind_dir: Union[Unset, str] = UNSET
    pressure_mb: Union[Unset, float] = UNSET
    pressure_in: Union[Unset, float] = UNSET
    precip_mm: Union[Unset, float] = UNSET
    precip_in: Union[Unset, float] = UNSET
    humidity: Union[Unset, int] = UNSET
    cloud: Union[Unset, int] = UNSET
    feelslike_c: Union[Unset, float] = UNSET
    feelslike_f: Union[Unset, float] = UNSET
    vis_km: Union[Unset, float] = UNSET
    vis_miles: Union[Unset, float] = UNSET
    uv: Union[Unset, float] = UNSET
    gust_mph: Union[Unset, float] = UNSET
    gust_kph: Union[Unset, float] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        from ..models.check_weather_using_get_response_200_current_condition import (
            CheckWeatherUsingGETResponse200CurrentCondition,
        )
        last_updated = self.last_updated
        last_updated_epoch = self.last_updated_epoch
        temp_c = self.temp_c
        temp_f = self.temp_f
        is_day = self.is_day
        condition: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.condition, Unset):
            condition = self.condition.to_dict()

        wind_mph = self.wind_mph
        wind_kph = self.wind_kph
        wind_degree = self.wind_degree
        wind_dir = self.wind_dir
        pressure_mb = self.pressure_mb
        pressure_in = self.pressure_in
        precip_mm = self.precip_mm
        precip_in = self.precip_in
        humidity = self.humidity
        cloud = self.cloud
        feelslike_c = self.feelslike_c
        feelslike_f = self.feelslike_f
        vis_km = self.vis_km
        vis_miles = self.vis_miles
        uv = self.uv
        gust_mph = self.gust_mph
        gust_kph = self.gust_kph

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if last_updated is not UNSET:
            field_dict["last_updated"] = last_updated
        if last_updated_epoch is not UNSET:
            field_dict["last_updated_epoch"] = last_updated_epoch
        if temp_c is not UNSET:
            field_dict["temp_c"] = temp_c
        if temp_f is not UNSET:
            field_dict["temp_f"] = temp_f
        if is_day is not UNSET:
            field_dict["is_day"] = is_day
        if condition is not UNSET:
            field_dict["condition"] = condition
        if wind_mph is not UNSET:
            field_dict["wind_mph"] = wind_mph
        if wind_kph is not UNSET:
            field_dict["wind_kph"] = wind_kph
        if wind_degree is not UNSET:
            field_dict["wind_degree"] = wind_degree
        if wind_dir is not UNSET:
            field_dict["wind_dir"] = wind_dir
        if pressure_mb is not UNSET:
            field_dict["pressure_mb"] = pressure_mb
        if pressure_in is not UNSET:
            field_dict["pressure_in"] = pressure_in
        if precip_mm is not UNSET:
            field_dict["precip_mm"] = precip_mm
        if precip_in is not UNSET:
            field_dict["precip_in"] = precip_in
        if humidity is not UNSET:
            field_dict["humidity"] = humidity
        if cloud is not UNSET:
            field_dict["cloud"] = cloud
        if feelslike_c is not UNSET:
            field_dict["feelslike_c"] = feelslike_c
        if feelslike_f is not UNSET:
            field_dict["feelslike_f"] = feelslike_f
        if vis_km is not UNSET:
            field_dict["vis_km"] = vis_km
        if vis_miles is not UNSET:
            field_dict["vis_miles"] = vis_miles
        if uv is not UNSET:
            field_dict["uv"] = uv
        if gust_mph is not UNSET:
            field_dict["gust_mph"] = gust_mph
        if gust_kph is not UNSET:
            field_dict["gust_kph"] = gust_kph

        return field_dict



    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.check_weather_using_get_response_200_current_condition import (
            CheckWeatherUsingGETResponse200CurrentCondition,
        )
        d = src_dict.copy()
        last_updated = d.pop("last_updated", UNSET)

        last_updated_epoch = d.pop("last_updated_epoch", UNSET)

        temp_c = d.pop("temp_c", UNSET)

        temp_f = d.pop("temp_f", UNSET)

        is_day = d.pop("is_day", UNSET)

        _condition = d.pop("condition", UNSET)
        condition: Union[Unset, CheckWeatherUsingGETResponse200CurrentCondition]
        if isinstance(_condition,  Unset):
            condition = UNSET
        else:
            condition = CheckWeatherUsingGETResponse200CurrentCondition.from_dict(_condition)




        wind_mph = d.pop("wind_mph", UNSET)

        wind_kph = d.pop("wind_kph", UNSET)

        wind_degree = d.pop("wind_degree", UNSET)

        wind_dir = d.pop("wind_dir", UNSET)

        pressure_mb = d.pop("pressure_mb", UNSET)

        pressure_in = d.pop("pressure_in", UNSET)

        precip_mm = d.pop("precip_mm", UNSET)

        precip_in = d.pop("precip_in", UNSET)

        humidity = d.pop("humidity", UNSET)

        cloud = d.pop("cloud", UNSET)

        feelslike_c = d.pop("feelslike_c", UNSET)

        feelslike_f = d.pop("feelslike_f", UNSET)

        vis_km = d.pop("vis_km", UNSET)

        vis_miles = d.pop("vis_miles", UNSET)

        uv = d.pop("uv", UNSET)

        gust_mph = d.pop("gust_mph", UNSET)

        gust_kph = d.pop("gust_kph", UNSET)

        check_weather_using_get_response_200_current = cls(
            last_updated=last_updated,
            last_updated_epoch=last_updated_epoch,
            temp_c=temp_c,
            temp_f=temp_f,
            is_day=is_day,
            condition=condition,
            wind_mph=wind_mph,
            wind_kph=wind_kph,
            wind_degree=wind_degree,
            wind_dir=wind_dir,
            pressure_mb=pressure_mb,
            pressure_in=pressure_in,
            precip_mm=precip_mm,
            precip_in=precip_in,
            humidity=humidity,
            cloud=cloud,
            feelslike_c=feelslike_c,
            feelslike_f=feelslike_f,
            vis_km=vis_km,
            vis_miles=vis_miles,
            uv=uv,
            gust_mph=gust_mph,
            gust_kph=gust_kph,
        )

        check_weather_using_get_response_200_current.additional_properties = d
        return check_weather_using_get_response_200_current

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
