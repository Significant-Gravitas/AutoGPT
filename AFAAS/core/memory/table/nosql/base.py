from __future__ import annotations

import abc
import datetime
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel

from AFAAS.app.sdk import forge_log
from AFAAS.core.configuration import (AFAASModel,
                                                         SystemSettings)

from ..base import AbstractTable

LOG = forge_log.ForgeLogger(__name__)


class BaseSQLTable(AbstractTable):
    def __init__(self) -> None:
        raise NotImplementedError()

    def add(self, value: dict) -> uuid.UUID:
        id = uuid.uuid4()
        value["id"] = id
        self.memory.add(key=id, value=value, table_name=self.table_name)
        return id


# TODO : Adopt Configurable ?
class BaseNoSQLTable(AbstractTable):
    class Key(TypedDict):
        primary_key: str
        secondary_key: str
        third_key: Optional[Any]

    secondary_key: Optional[str]

    def deserialize(dct):
        if "_type_" in dct:
            parts = dct["_type_"].rsplit(".", 1)
            module = __import__(parts[0])
            class_ = getattr(module, parts[1])
            obj = class_()
            for key, value in dct.items():
                if key != "_type_":
                    setattr(obj, key, value)
            return obj
        return dct

    # NOTE : Move to marshmallow ?!?
    # https://marshmallow.readthedocs.io/en/stable/quickstart.html#serializing-objects-dumping
    @classmethod
    def serialize_value(self, value) -> dict:
        stack = [(value, {}, None)]
        root_dict = stack[0][1]
        count = 0
        while stack:
            count += 1
            # print(f"\n\n\ncount : {count}")
            curr_obj, parent_dict, key = stack.pop()

            if isinstance(curr_obj, (str, int, float, bool, type(None))):
                serialized_value = curr_obj
            elif isinstance(curr_obj, uuid.UUID):
                serialized_value = str(curr_obj)
            elif isinstance(curr_obj, Path):
                serialized_value = str(curr_obj)
            elif isinstance(curr_obj, dict):
                new_dict = {}
                for k, v in curr_obj.items():
                    stack.append((v, new_dict, k))
                serialized_value = new_dict
            elif isinstance(curr_obj, (list, tuple)):
                new_list = [None] * len(curr_obj)
                for idx, val in enumerate(curr_obj):
                    stack.append((val, new_list, idx))
                serialized_value = new_list
            elif isinstance(curr_obj, set):
                new_list = [None] * len(curr_obj)
                for idx, val in enumerate(list(curr_obj)):
                    stack.append((val, new_list, idx))
                serialized_value = new_list
            elif isinstance(curr_obj, BaseModel):
                serialized_value = curr_obj.dict()
            elif hasattr(curr_obj, "__dict__"):
                new_dict = {}
                new_dict[
                    "_type_"
                ] = f"{curr_obj.__class__.__module__}.{curr_obj.__class__.__name__}"
                for attr, attr_value in curr_obj.__dict__.items():
                    if not (
                        attr.startswith("_")
                        or attr in value.__class__.SystemSettings.Config.default_exclude
                    ):
                        stack.append((attr_value, new_dict, attr))
                serialized_value = new_dict
            else:
                serialized_value = str(curr_obj)

            if key is not None:
                parent_dict[key] = serialized_value

            # if (key) :
            #     print( "key = " , key )
            # if parent_dict :
            #     print( "parent_dict = " ,parent_dict )
            # if curr_obj :
            #     print( "curr_obj = " , curr_obj  )

        return parent_dict

    def add(self, value: dict, id: str = str(uuid.uuid4())) -> uuid.UUID:
        # Serialize non-serializable objects
        if isinstance(value, AFAASModel):
            value = value.dict_memory()
        else:
            self.memory._logger.warning("Class not hinheriting from AFAASModel")
            value = self.__class__.serialize_value(value)

        # Assigning primary key
        key = {"primary_key": str(id)}
        value[self.primary_key] = str(id)

        if hasattr(self, "secondary_key") and self.secondary_key in value:
            key["secondary_key"] = str(value[self.secondary_key])
            value[self.secondary_key] = str(value[self.secondary_key])

        self.memory._logger.debug(
            "add new " + str(self.__class__.__name__) + "with keys " + str(key)
        )
        self.memory._logger.debug(
            "add new " + str(self.__class__.__name__) + "with values " + str(value)
        )

        self.memory.add(key=key, value=value, table_name=self.table_name)
        return id

    @abc.abstractmethod
    def update(self, key: BaseNoSQLTable.Key, value: dict):
        # Serialize non-serializable objects
        if isinstance(value, BaseModel):
            value = value.dict()

        value["modified_at"]: datetime.datetime = datetime.datetime.now()
        value = self.__class__.serialize_value(value)

        # key = {"primary_key": id}
        # if hasattr(self, "secondary_key") and self.secondary_key in value:
        #     key["secondary_key"] = value[self.secondary_key]

        self.memory._logger.debug(
            "Update new " + str(self.__class__.__name__) + "with keys " + str(key)
        )
        self.memory._logger.debug(
            "Update new " + str(self.__class__.__name__) + "with values " + str(value)
        )

        self.memory.update(key=key, value=value, table_name=self.table_name)

    @abc.abstractmethod
    def get(self, key: BaseNoSQLTable.Key) -> Any:
        return self.memory.get(key=key, table_name=self.table_name)

    @abc.abstractmethod
    def delete(self, key: BaseNoSQLTable.Key):
        # key = {"primary_key": id}
        # if hasattr(self, "secondary_key") and self.secondary_key:
        #     key["secondary_key"] = self.secondary_key
        self.memory.delete(key=key, table_name=self.table_name)

    def list(
        self,
        filter: AbstractTable.FilterDict = {},
        order_column: Optional[str] = None,
        order_direction: Literal["asc", "desc"] = "desc",
    ) -> list[dict[str, Any]]:
        """
        Retrieve a filtered and optionally ordered list of items from the table.

        Parameters:
            filter (FilterDict, optional): A dictionary containing the filter conditions.
                The keys in the filter dictionary represent the column names, and the values
                are dictionaries containing the 'value' and 'operator' keys.
                'value': The value used for comparison in the filter.
                'operator': The operator to use for comparison. It can be one of the operators defined
                            in BaseTable.Operators enum or a custom callable.
                            - For predefined operators, use BaseTable.Operators, e.g., BaseTable.Operators.GREATER_THAN.
                            - For custom operators, provide a callable that takes two arguments
                            and returns a bool indicating the result of the comparison.

            order_column (str, optional): The column name to use for sorting the results.
                Default: None.

            order_direction (str, optional): The order direction for sorting the results.
                Can be 'asc' (ascending) or 'desc' (descending). Default: 'desc'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the filtered items that
                                match the specified conditions.

        Example:
            Suppose we have a BaseTable instance with the following data in the memory:

            data_list = [
                {'name': 'John', 'age': 25, 'city': 'New York'},
                {'name': 'Alice', 'age': 30, 'city': 'Los Angeles'},
                {'name': 'Bob', 'age': 22, 'city': 'Chicago'},
                {'name': 'Eve', 'age': 35, 'city': 'San Francisco'}
            ]

            # Example 1: Using BaseTable.Operators.GREATER_THAN for age greater than 25
            filter_dict = {'age': {'value': 25, 'operator': BaseTable.Operators.GREATER_THAN}}
            result = base_table.list(filter_dict)
            # Output: [{'name': 'Alice', 'age': 30, 'city': 'Los Angeles'},
            #          {'name': 'Eve', 'age': 35, 'city': 'San Francisco'}]

            # Example 2: Using custom operator for a specific filter
            def custom_comparison(value, filter_value):
                return len(value['city']) > len(filter_value)

            filter_dict = {'city': {'value': 'Chicago', 'operator': custom_comparison}}
            result = base_table.list(filter_dict)
            # Output: [{'name': 'John', 'age': 25, 'city': 'New York'},
            #          {'name': 'Alice', 'age': 30, 'city': 'Los Angeles'}]

            # Example 3: Using multiple filters with predefined and custom operators
            filter_dict = {
                'age': {'value': 30, 'operator': BaseTable.Operators.GREATER_THAN_OR_EQUAL},
                'city': {'value': 'New York', 'operator': BaseTable.Operators.NOT_EQUAL_TO},
                'name': {'value': 'Bob', 'operator': custom_comparison}
            }
            result = base_table.list(filter_dict)
            # Output: [{'name': 'Alice', 'age': 30, 'city': 'Los Angeles'}]
        """
        data_list = self.memory.list(table_name=self.table_name)
        filtered_data_list: list = []

        LOG.notice("May need to be moved to JSONFileMemory")

        for data in data_list:
            remove_entry = False
            for filter_column_name, filters in filter.items():
                value_to_filter = data.get(filter_column_name)
                # NOTE: May be this test & the else need to be removed
                if value_to_filter is not None:
                    for filter_data in filters:
                        filter_value = filter_data["value"]
                        filter_operator = filter_data["operator"]
                        if isinstance(filter_operator, AbstractTable.Operators):
                            comparison_function = filter_operator.value
                        elif callable(filter_operator):
                            comparison_function = filter_operator
                        else:
                            raise ValueError(
                                f"Invalid comparison operator: {filter_operator}"
                            )
                        if not comparison_function(value_to_filter, filter_value):
                            remove_entry = True
                else:
                    remove_entry = True
            if not remove_entry:
                filtered_data_list.append(data)

        if order_column:
            filtered_data_list.sort(
                key=lambda x: x[order_column], reverse=order_direction == "desc"
            )

        return filtered_data_list
