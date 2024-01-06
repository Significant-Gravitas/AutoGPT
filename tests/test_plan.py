from __future__ import annotations

import copy
import json
import uuid

import pytest

from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import Task

from .dataset.plan_familly_dinner import (
    plan_step_0,
    plan_step_1,
    plan_step_2a,
    plan_step_2b,
    plan_step_3,
    plan_step_4,
    plan_step_5,
    plan_step_6,
    plan_step_7,
    plan_step_8,
    plan_step_9,
    plan_step_10,
    plan_step_11,
    task_awaiting_preparation,
    task_ready_all_predecessors_done,
    task_ready_all_subtasks_done,
    task_ready_no_predecessors_or_subtasks,
    task_with_mixed_predecessors,
    task_with_ongoing_subtasks,
    task_with_unmet_predecessors,
)

# @pytest.mark.parametrize(
#     "plan_step, expected_task_ids",
#     [
#         # Initial state, where tasks 5, 15, 25, 35, and 300.1.1 are ready and have no subtasks
#         (plan_step_0, ["5", "15", "25", "35", "300.1.1"]),
#         (plan_step_1, ["45", "106", "300.1.2"]),
#         # After subtask 300.1.2 is done, task 300.1 can be marked as done, and tasks 107, 200, 200.1 become ready
#         (plan_step_2a, ["45", "106", "107", "200", "200.1"]),
#         # After tasks 106, 200.1 are done, tasks 200.2, 200.3, 300.2, 300.3.1, and 300.3.2 become ready
#         (
#             plan_step_3,
#             ["45", "107", "200", "200.2", "200.3", "300.2", "300.3.1", "300.3.2"],
#         ),
#         # After tasks 200.2, 200.3, 300.3.1, 300.3.2 are done, tasks 200.4, 300.3.3, and 300.3 become ready
#         (plan_step_4, ["45", "107", "200", "200.4", "300.3.3", "300.3"]),
#         # After tasks 200.4, 300.3.3, 107, 200 are done, tasks 108, 300.4 become ready
#         (plan_step_5, ["45", "108", "300.3", "300.4"]),
#         # After tasks 108, 300.3 are done, tasks 201, 300.5 become ready
#         (plan_step_6, ["45", "201", "300.4", "300.5"]),
#         # After tasks 201, 300.4 are done, task 300.6 becomes ready
#         (plan_step_7, ["45", "300.5", "300.6"]),
#         # After task 300.5 is done, task 300.6 remains ready
#         (plan_step_8, ["45", "300.6"]),
#         # After task 300.6 is done, no more tasks are ready
#         (plan_step_9, ["45"]),
#         # After task 45 is done, no more tasks are ready
#         (plan_step_10, []),
#     ],
# )
# def test_get_ready_leaf_tasks(plan_step, expected_task_ids):
#     ready_tasks = plan_step.get_ready_leaf_tasks()
#     ready_task_ids = [task.task_id for task in ready_tasks]
#     assert set(ready_task_ids) == set(expected_task_ids)


# @pytest.mark.parametrize(
#     "plan_step, expected_task_id",
#     [
#         (plan_step_0, "5"),  # "5. Buy Groceries" is ready
#         (plan_step_1(), "45"),  # "45. Set the Mood for Dinner" is ready
#         (plan_step_2a(), "106"),  # "106. Set the Table" is ready
#         (plan_step_3(), "300.1"),
#         (plan_step_4(), "200.4"),
#         (plan_step_5(), "108"),  # "108. Serve Salad" is ready
#         (plan_step_6(), "201"),  # "201. Serve Main Course" is ready
#         (plan_step_7(), "300.5"),  # "300.5. Cool the Bread" is ready
#         (plan_step_8(), "300.6"),  # "300.6. Serve Banana Bread" is ready
#         (plan_step_9(), "45"),  # "45. Set the Mood for Dinner" is ready
#         (plan_step_10(), None),  # No task is ready
#     ],
# )
# def test_get_first_ready_task(plan_step, expected_task_id):
#     first_ready_task = plan_step.get_first_ready_task()
#     if expected_task_id is None:
#         assert first_ready_task is None
#     else:
#         assert first_ready_task.task_id == expected_task_id


# def test_dict_memory(plan_step_2b: Plan):
#     # Test case: Ensure it returns a dictionary representation of the object with custom encoders applied.
#     dict_memory_output = plan_step_2b.dict_memory()
#     assert isinstance(dict_memory_output, dict)
#     # Add more specific checks if you have custom encoder expectations

#     # Test case: Check behavior with different dumps_kwargs inputs.
#     # Example: passing 'exclude_none=True' to dict_memory
#     dict_memory_exclude_none = plan_step_2b.dict_memory(exclude_none=True)
#     assert isinstance(dict_memory_exclude_none, dict)
#     # Add more specific checks as needed


# def test_apply_custom_encoders(plan_step_2b: Plan):
#     # Prepare a sample data dictionary for testing
#     sample_data = {"test_field": uuid.uuid4()}  # Add more fields as necessary
#     # Test case: Verify that custom encoders are correctly applied
#     encoded_data = plan_step_2b._apply_custom_encoders(sample_data)
#     assert isinstance(
#         encoded_data["test_field"], str
#     )  # Assuming uuid is encoded to str

#     # Test case: Test with various data types that match the custom encoders
#     # Add more tests with different data types as per your custom encoders


# def test_dict_method(plan_step_2b: Plan):
#     # Test case: Confirm dictionary representation of the object, excluding default fields when include_all is False.
#     dict_output = plan_step_2b.dict()
#     assert isinstance(dict_output, dict)
#     # Check for the absence of default excluded fields

#     # Test case: Test with include_all set to True.
#     dict_output_include_all = plan_step_2b.dict(include_all=True)
#     assert isinstance(dict_output_include_all, dict)
#     # Add specific checks for fields that should be included when include_all is True


# def test_json_method(plan_step_2b: Plan):
#     # Test case: Ensure it serializes the object to JSON format correctly.
#     json_output = plan_step_2b.json()
#     assert isinstance(json_output, str)
#     # You can also parse the JSON and perform more detailed checks on the structure

#     # Test case: Test with different arguments and keyword arguments.
#     json_output_with_args = plan_step_2b.json(exclude={"task_goal"})
#     assert "task_goal" not in json.loads(json_output_with_args)


# def test_prepare_values_before_serialization(plan_step_2b: Plan):
#     # Test case: Check if the method prepares values correctly before serialization.
#     # This depends on what prepare_values_before_serialization actually does.
#     # Assuming it modifies some values in the object:
#     original_state = plan_step_2b.dict()
#     plan_step_2b.prepare_values_before_serialization()
#     modified_state = plan_step_2b.dict()
#     assert original_state != modified_state
#     # Add more specific assertions based on expected behavior


# def test_str_and_repr_methods(plan_step_2b: Plan):
#     # Test case: Verify that __str__ and __repr__ return a string representation of the object.
#     str_output = str(plan_step_2b)
#     repr_output = repr(plan_step_2b)
#     assert isinstance(str_output, str)
#     assert isinstance(repr_output, str)
#     # Optionally, add more detailed checks if you expect specific formats


def test_generate_uuid():
    # Test case: Check if it generates a unique UUID each time it's called.
    uuid1 = Task.generate_uuid()
    uuid2 = Task.generate_uuid()
    assert isinstance(uuid1, str) and isinstance(uuid2, str)
    assert uuid1 != uuid2
