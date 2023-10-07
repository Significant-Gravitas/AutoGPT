"""
Small pandas interface for CSV and other tabulated data
"""
from typing import List
import pandas
import csv
from forge.sdk.memory.memstore_tools import add_ability_memory

from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)

# get separator for readlines data
def get_sep(tfile_readlines):
    sn = csv.Sniffer()
    delim = sn.sniff(tfile_readlines[0].decode()).delimiter
    return delim


@ability(
    name="csv_get_columns",
    description=f"Get the names, if any, of the columns in a CSV",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        }
    ],
    output_type="str"
)
async def csv_get_columns(
    agent,
    task_id: str,
    file_name: str
) -> List[str]:
    file_readlines = agent.workspace.readlines(task_id=task_id, path=file_name)
    file_sep = get_sep(file_readlines)

    gcwd = agent.workspace.get_cwd_path(task_id)
    df = pandas.read_csv(f"{gcwd}/{file_name}", sep=file_sep)

    return list(df.columns)

@ability(
    name="csv_group_by_sum",
    description=f"Group two columns in CSV and get a sum",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        },
        {
            "name": "column_1",
            "description": "Primary column to group",
            "type": "string",
            "required": True
        },
        {
            "name": "column_2",
            "description": "Secondary column to group with Primary",
            "type": "string",
            "required": True
        }
    ],
    output_type="str"
)

async def csv_group_by_sum(
    agent,
    task_id: str,
    file_name: str,
    column_1: str,
    column_2: str
) -> str:
    file_readlines = agent.workspace.readlines(task_id=task_id, path=file_name)
    file_sep = get_sep(file_readlines)

    gcwd = agent.workspace.get_cwd_path(task_id)
    df = pandas.read_csv(f"{gcwd}/{file_name}", sep=file_sep)

    cat_sum = df.groupby(column_1).agg({column_2: "sum"})

    return cat_sum.to_string() 