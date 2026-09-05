from typing import Optional

from e2b_code_interpreter.charts import Chart as E2BExecutionResultChart
from pydantic import BaseModel, Field, JsonValue

MAIN_RESULT_DESCRIPTION = (
    "The main result from the code execution (the script's final "
    "expression). Its `json` sub-field is ONLY populated when the "
    "result is a dict/object/map — bare lists, strings, and "
    "numbers land in `text` as a string instead. To pass "
    "structured data downstream via `main_result_#_json_#_<key>` "
    "links, end the script with a key-value structure in the "
    "script's language (e.g. `{'items': my_list}` in Python, "
    "`({items: myList})` in JavaScript)."
)


class MainCodeExecutionResult(BaseModel):
    """
    *Pydantic model mirroring `e2b_code_interpreter.Result`*

    Represents the data to be displayed as a result of executing a cell in a Jupyter notebook.
    The result is similar to the structure returned by ipython kernel: https://ipython.readthedocs.io/en/stable/development/execution.html#execution-semantics

    The result can contain multiple types of data, such as text, images, plots, etc. Each type of data is represented
    as a string, and the result can contain multiple types of data. The display calls don't have to have text representation,
    for the actual result the representation is always present for the result, the other representations are always optional.
    """  # noqa

    class Chart(BaseModel, E2BExecutionResultChart):
        pass

    text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    svg: Optional[str] = None
    png: Optional[str] = None
    jpeg: Optional[str] = None
    pdf: Optional[str] = None
    latex: Optional[str] = None
    json_data: Optional[JsonValue] = Field(None, alias="json")
    javascript: Optional[str] = None
    data: Optional[dict] = None
    chart: Optional[Chart] = None
    extra: Optional[dict] = None
    """Extra data that can be included. Not part of the standard types."""


class CodeExecutionResult(MainCodeExecutionResult):
    __doc__ = MainCodeExecutionResult.__doc__

    is_main_result: bool = False
    """Whether this data is the main result of the cell. Data can be produced by display calls of which can be multiple in a cell."""  # noqa
