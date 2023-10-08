"""
Small pandas interface for CSV and other tabulated data
"""
from typing import List, Any
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

    if delim == "t" or delim == "n":
        delim = f"\{delim}"
    
    return delim

def load_csv(agent, task_id, file_name) -> Any:
    """
    Load CSV file in CWD
    """
    df = None
    if ".csv" in file_name or ".CSV" in file_name:
        file_readlines = agent.workspace.readlines(task_id=task_id, path=file_name)
        file_sep = get_sep(file_readlines)

        gcwd = agent.workspace.get_cwd_path(task_id)
        df = pandas.read_csv(f"{gcwd}/{file_name}", sep=file_sep)
    
    return df

@ability(
    name="csv_get_columns",
    description="Gets a list of column names in a CSV",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        }
    ],
    output_type="List"
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

    return df.columns.to_list()

@ability(
    name="get_amount_rows",
    description="Get the amount of rows",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        }
    ],
    output_type="int"
)
async def get_amount_rows(
    agent,
    task_id: str,
    file_name: str) -> int:

    df = load_csv(agent, task_id, file_name)
    if df:
        try:
            row_amt = len(df.index)
        except Exception as err:
            logger.error(f"getting row value failed: {err}")
            raise err
        
        return row_amt
    else:
        return "No CSV file found"

@ability(
    name="get_column_value",
    description="Get the value of a column at a certain row",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        },
        {
            "name": "column",
            "description": "Name of the column",
            "type": "string",
            "required": True
        },
        {
            "name": "row_idx",
            "description": "Row number starting from 0",
            "type": "int",
            "required": True
        }
    ],
    output_type="Any"
)
async def get_column_value(
    agent,
    task_id: str,
    file_name: str,
    column: str,
    row_idx: str
) -> Any:
    df = load_csv(agent, task_id, file_name)
    if df:
        try:
            row_val = df.iloc[row_idx][column]
        except Exception as err:
            logger.error(f"getting row value failed: {err}")
            raise err
        
        return row_val
    else:
        return "No CSV file found"

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

@ability(
    name="csv_check_column",
    description=f"Boolean check to see if known CSV column has row value",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        },
        {
            "name": "column",
            "description": "Primary column",
            "type": "string",
            "required": True
        },
        {
            "name": "row_value",
            "description": "Row value to check",
            "type": "string",
            "required": True
        }
    ],
    output_type="bool"
)
async def csv_check_column(
    agent, 
    task_id: str, 
    file_name: str, 
    column: str, 
    row_value: str) -> bool:
    
    file_readlines = agent.workspace.readlines(task_id=task_id, path=file_name)
    file_sep = get_sep(file_readlines)

    gcwd = agent.workspace.get_cwd_path(task_id)
    df = pandas.read_csv(f"{gcwd}/{file_name}", sep=file_sep)

    if column in df.columns:
        df.loc[df[column] == row_value]
        if not df.empty:
            return True
        
    return False

@ability(
    name="csv_merge",
    description=f"Merge two CSV files by a column present in both",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the new file to hold merged CSV data",
            "type": "string",
            "required": True
        },
        {
            "name": "file_1",
            "description": "First file to merge",
            "type": "string",
            "required": True
        },
        {
            "name": "file_2",
            "description": "Second file to merge",
            "type": "string",
            "required": True
        },
        {
            "name": "on_column",
            "description": "Which column should both files be merged upon",
            "type": "string",
            "required": True
        }
    ],
    output_type="none"
)
async def csv_merge(
    agent,
    task_id: str,
    file_name: str,
    file_1: str,
    file_2: str,
    on_column: str 
    ) -> None:

    try:
        gcwd = agent.workspace.get_cwd_path(task_id)

        file_readlines_1 = agent.workspace.readlines(task_id=task_id, path=file_1)
        file_sep_1 = get_sep(file_readlines_1)

        file_readlines_2 = agent.workspace.readlines(task_id=task_id, path=file_2)
        file_sep_2 = get_sep(file_readlines_2)

        df1 = pandas.read_csv(f"{gcwd}/{file_1}", sep=file_sep_1)
        df2 = pandas.read_csv(f"{gcwd}/{file_2}", sep=file_sep_2)
        
        df = pandas.merge(df1, df2, on=on_column, how="inner")

        # cf_paths = []
        # for cfile in combine_files:
        #     cf_paths.append(f"{gcwd}/{cfile}")
        
        # df = pandas.concat(
        #     map(pandas.read_csv, cf_paths), ignore_index=True) 
        
        
        df.to_csv(f"{gcwd}/{file_name}")
    except Exception as err:
        logger.error(f"csv_comine failed: {err}")

@ability(
    name="csv_sort_by_column",
    description=f"Sort CSV by a certain column",
    parameters=[
        {
            "name": "input_file",
            "description": "Name of the input file",
            "type": "string",
            "required": True
        },
        {
            "name": "column",
            "description": "Column to sort by",
            "type": "string",
            "required": True
        },
        {
            "name": "output_file",
            "description": "Name of the output file",
            "type": "string",
            "required": True
        }
    ],
    output_type="bool"
)
async def csv_sort_by_column(
    agent, 
    task_id: str,
    input_file: str, 
    column: str,
    output_file: str) -> bool:

    try:
        gcwd = agent.workspace.get_cwd_path(task_id)

        file_readlines = agent.workspace.readlines(
            task_id=task_id, path=input_file)
        file_sep = get_sep(file_readlines)

        df = pandas.read_csv(
            f"{gcwd}/{input_file}", 
            sep=file_sep
        )

        sorted_df = df.sort_values(by=column)
        sorted_df.to_csv(output_file, index=False)
    except Exception as err:
        logger.error(f"csv_sort_by_column failed: {err}")
        return False
    
    return True

@ability(
    name="csv_add_column_data",
    description="Add data to column or creates column in CSV and adds data",
    parameters=[
        {
            "name": "input_file",
            "description": "Name of the input file",
            "type": "string",
            "required": True
        },
        {
            "name": "output_file",
            "description": "Name of the output file",
            "type": "string",
            "required": True
        },
        {
            "name": "column",
            "description": "Column to sort by",
            "type": "string",
            "required": True
        },
        {
            "name": "value",
            "description": "Value to add to column",
            "type": "Any",
            "required": True
        },
        {
            "name": "row_index",
            "description": "the index in the column to add the value",
            "type": "int",
            "required": True
        }
    ],
    output_type="bool"
)
async def csv_add_column_data(
    agent,
    task_id: str,
    input_file: str,
    output_file: str, 
    column: str,
    value: Any,
    row_index: int) -> bool:

    try:
        gcwd = agent.workspace.get_cwd_path(task_id)

        file_readlines = agent.workspace.readlines(
            task_id=task_id, path=input_file)
        file_sep = get_sep(file_readlines)

        df = pandas.read_csv(
            f"{gcwd}/{input_file}", 
            sep=file_sep
        )

        df.loc[row_index, column] = value
        df.to_csv(f"{gcwd}/{output_file}", index=False)
    except Exception as err:
        logger.error(f"csv_add_column_data failed: {err}")
        return False
    
    return True