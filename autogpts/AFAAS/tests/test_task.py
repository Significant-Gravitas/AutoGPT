import pytest
import json
import uuid
import copy

from autogpts.AFAAS.app.lib.task.plan import Plan
from autogpts.AFAAS.app.lib.task.meta import TaskStatusList
from autogpts.AFAAS.app.lib.task.task import Task

plan_prepare_dinner = Plan(task_id="1000", task_goal="1000. Prepare Dinner for Family")
task_buy_groceries = Task(task_id="5", task_goal="5. Buy Groceries")
task_buy_groceries.state = TaskStatusList.READY
task_clean_kitchen = Task(task_id="15", task_goal="15. Clean Kitchen")
task_clean_kitchen.state = TaskStatusList.READY
task_choose_music = Task(task_id="25", task_goal="25. Choose Dinner Music")
task_choose_music.state = TaskStatusList.READY
task_decorate_dining_room = Task(task_id="35", task_goal="35. Decorate Dining Room")
task_decorate_dining_room.state = TaskStatusList.READY
task_set_mood = Task(task_id="45", task_goal="45. Set the Mood for Dinner")
task_set_mood.add_predecessor(task_choose_music)
task_set_mood.add_predecessor(task_decorate_dining_room)

task_set_table = Task(task_id="110", task_goal="110. Set the Table")
task_set_table.add_predecessor(task_clean_kitchen)
task_make_salad = Task(task_id="120", task_goal="120. Make Salad")
task_make_salad.add_predecessor(task_set_table)
task_serve_salad = Task(task_id="390", task_goal="390. Serve Salad")
task_serve_salad.add_predecessor(task_make_salad)
task_prepare_main_course = Task(task_id="130", task_goal="130. Prepare Main Course")
task_prepare_main_course.add_predecessor(task_set_table)
task_serve_dinner = Task(task_id="400", task_goal="400. Serve Main Course")
task_serve_dinner.add_predecessor(task_prepare_main_course)
task_serve_dinner.add_predecessor(task_serve_salad)
task_make_banana_bread = Task(task_id="1000", task_goal="500. Make Banana Bread")

# Subtasks for 'Make Banana Bread'
task_gather_ingredients = Task(task_id="50", task_goal="50. Gather Ingredients")
task_gather_ingredients.add_predecessor(task_buy_groceries)
task_gather_ingredients.add_predecessor(task_clean_kitchen)
task_preheat_oven = Task(task_id="310", task_goal="310. Preheat Oven")
task_preheat_oven.add_predecessor(task_gather_ingredients)
task_prepare_baking_pan = Task(task_id="320", task_goal="320. Prepare Baking Pan")
task_prepare_baking_pan.add_predecessor(task_gather_ingredients)
task_mix_ingredients = Task(task_id="440", task_goal="440. Mix Ingredients")
task_mix_ingredients.add_predecessor(task_buy_groceries)
task_mix_ingredients.add_predecessor(task_clean_kitchen)
task_mix_ingredients.add_predecessor(task_gather_ingredients)
task_bake_bread = Task(task_id="450", task_goal="450. Bake the Bread")
task_bake_bread.add_predecessor(task_mix_ingredients)
task_cool_bread = Task(task_id="490", task_goal="490. Cool the Bread")
task_cool_bread.add_predecessor(task_bake_bread)
task_serve_bread = Task(task_id="500", task_goal="400. Serve Banana Bread")
task_serve_bread.add_predecessor(task_cool_bread)
task_serve_bread.add_predecessor(task_serve_dinner)

# Subtasks for 'Mix Ingredients'
subtask_measure_flour = Task(task_id="351", task_goal="351. Measure Flour")
subtask_measure_flour.add_predecessor(task_gather_ingredients)
subtask_mash_bananas = Task(task_id="352", task_goal="352. Mash Bananas")
subtask_mash_bananas.add_predecessor(task_gather_ingredients)
subtask_combine_wet_ingredients = Task(task_id="353", task_goal="353. Combine Wet Ingredients")
subtask_combine_wet_ingredients.add_predecessor(subtask_mash_bananas)
subtask_combine_wet_ingredients.add_predecessor(subtask_measure_flour)

plan_prepare_dinner.add_task(task_make_banana_bread)
plan_prepare_dinner.add_task(task_set_table)
plan_prepare_dinner.add_task(task_make_salad)
plan_prepare_dinner.add_task(task_serve_salad)
plan_prepare_dinner.add_task(task_prepare_main_course)
plan_prepare_dinner.add_task(task_serve_dinner)
plan_prepare_dinner.add_task(task_buy_groceries)
plan_prepare_dinner.add_task(task_clean_kitchen)
plan_prepare_dinner.add_task(task_choose_music)
plan_prepare_dinner.add_task(task_decorate_dining_room)
plan_prepare_dinner.add_task(task_set_mood)
task_make_banana_bread.add_task(task_gather_ingredients)
task_make_banana_bread.add_task(task_preheat_oven)
task_make_banana_bread.add_task(task_prepare_baking_pan)
task_make_banana_bread.add_task(task_mix_ingredients)
task_make_banana_bread.add_task(task_bake_bread)
task_make_banana_bread.add_task(task_cool_bread)
task_make_banana_bread.add_task(task_serve_bread)
task_mix_ingredients.add_task(subtask_measure_flour)
task_mix_ingredients.add_task(subtask_mash_bananas)
task_mix_ingredients.add_task(subtask_combine_wet_ingredients)

@pytest.fixture
def plan_step_0():
    # Task 'task_prepare_dinner' has multiple subtasks
    return copy.deepcopy(plan_prepare_dinner)

@pytest.fixture
def plan_step_1():
    t : Plan = plan_step_0()
    t.get_task(task_id="5").state = TaskStatusList.DONE
    t.get_task(task_id="15").state = TaskStatusList.DONE
    t.get_task(task_id="25").state = TaskStatusList.DONE
    t.get_task(task_id="35").state = TaskStatusList.DONE
    t.get_task(task_id="45").state = TaskStatusList.READY
    t.get_task(task_id="50").state = TaskStatusList.READY
    t.get_task(task_id="110").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_2a():
    t : Plan = plan_step_1()
    t.get_task(task_id="110").state = TaskStatusList.DONE
    t.get_task(task_id="120").state = TaskStatusList.READY
    t.get_task(task_id="130").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t


@pytest.fixture
def plan_step_2b():
    t : Plan = plan_step_1()
    t.get_task(task_id="45").state = TaskStatusList.DONE
    t.get_task(task_id="50").state = TaskStatusList.DONE
    t.get_task(task_id="310").state = TaskStatusList.READY
    t.get_task(task_id="320").state = TaskStatusList.READY
    t.get_task(task_id="351").state = TaskStatusList.READY
    t.get_task(task_id="352").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_3a():
    t : Plan = plan_step_2b()
    t.get_task(task_id="120").state = TaskStatusList.DONE
    t.get_task(task_id="130").state = TaskStatusList.DONE
    t.get_task(task_id="390").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t


@pytest.fixture
def plan_step_3b():
    t : Plan = plan_step_3a()
    t.get_task(task_id="310").state = TaskStatusList.DONE
    t.get_task(task_id="320").state = TaskStatusList.DONE
    t.get_task(task_id="351").state = TaskStatusList.DONE
    t.get_task(task_id="352").state = TaskStatusList.DONE
    t.get_task(task_id="353").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_4a():
    t : Plan = plan_step_3b()
    t.get_task(task_id="390").state = TaskStatusList.DONE
    t.get_task(task_id="400").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_4b():
    t : Plan = plan_step_4a()
    t.get_task(task_id="353").state = TaskStatusList.DONE
    t.get_task(task_id="440").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_5a():
    t : Plan = plan_step_4b()
    t.get_task(task_id="400").state = TaskStatusList.DONE
    return t

@pytest.fixture
def plan_step_5b():
    t : Plan = plan_step_5a()
    t.get_task(task_id="440").state = TaskStatusList.DONE
    t.get_task(task_id="450").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_6():
    t : Plan = plan_step_5b()
    t.get_task(task_id="450").state = TaskStatusList.DONE
    t.get_task(task_id="490").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_7():
    t : Plan = plan_step_6()
    t.get_task(task_id="490").state = TaskStatusList.DONE
    t.get_task(task_id="500").state = TaskStatusList.READY
    # Task 'task_prepare_dinner' has multiple subtasks
    return t

@pytest.fixture
def plan_step_8():
    t : Plan = plan_step_7()
    t.get_task(task_id="500").state = TaskStatusList.DONE
    # Task 'task_prepare_dinner' has multiple subtasks
    return t


plan_steps = (
    plan_step_0,
    plan_step_1,
    plan_step_2a,
    plan_step_2b,
    plan_step_3a,
    plan_step_3b,
    plan_step_4a,
    plan_step_4b,
    plan_step_5a,
    plan_step_5b,
    plan_step_6,
    plan_step_7,
    plan_step_8,
)

@pytest.fixture
def task_ready_no_predecessors_or_subtasks(plan_step_0: Plan):
    # Task 'task_buy_groceries' has no predecessors or subtasks and is ready
    return plan_step_0.get_task(task_id="5")

@pytest.fixture
def task_ready_all_predecessors_done(plan_step_2b: Plan):
    # Task 'task_set_mood' with all predecessors done
    return plan_step_2b.get_task(task_id="45")

@pytest.fixture
def task_ready_all_subtasks_done(plan_step_8: Plan):
    # Task 'plan_prepare_dinner' with all subtasks done
    return plan_step_8.get_task(task_id="1000")

@pytest.fixture
def task_with_mixed_predecessors(plan_step_7: Plan):
    # Task 'task_serve_bread' with some predecessors done and some not
    return plan_step_7.get_task(task_id="500")



def test_dict_memory(plan_step_2b: Plan):
    # Test case: Ensure it returns a dictionary representation of the object with custom encoders applied.
    dict_memory_output = plan_step_2b.dict_memory()
    assert isinstance(dict_memory_output, dict)
    # Add more specific checks if you have custom encoder expectations

    # Test case: Check behavior with different dumps_kwargs inputs.
    # Example: passing 'exclude_none=True' to dict_memory
    dict_memory_exclude_none = plan_step_2b.dict_memory(exclude_none=True)
    assert isinstance(dict_memory_exclude_none, dict)
    # Add more specific checks as needed

def test_apply_custom_encoders(plan_step_2b: Plan):
    # Prepare a sample data dictionary for testing
    sample_data = {"test_field": uuid.uuid4()}  # Add more fields as necessary
    # Test case: Verify that custom encoders are correctly applied
    encoded_data = plan_step_2b._apply_custom_encoders(sample_data)
    assert isinstance(encoded_data['test_field'], str)  # Assuming uuid is encoded to str

    # Test case: Test with various data types that match the custom encoders
    # Add more tests with different data types as per your custom encoders

def test_dict_method(plan_step_2b: Plan):
    # Test case: Confirm dictionary representation of the object, excluding default fields when include_all is False.
    dict_output = plan_step_2b.dict()
    assert isinstance(dict_output, dict)
    # Check for the absence of default excluded fields

    # Test case: Test with include_all set to True.
    dict_output_include_all = plan_step_2b.dict(include_all=True)
    assert isinstance(dict_output_include_all, dict)
    # Add specific checks for fields that should be included when include_all is True

def test_json_method(plan_step_2b: Plan):
    # Test case: Ensure it serializes the object to JSON format correctly.
    json_output = plan_step_2b.json()
    assert isinstance(json_output, str)
    # You can also parse the JSON and perform more detailed checks on the structure

    # Test case: Test with different arguments and keyword arguments.
    json_output_with_args = plan_step_2b.json(exclude={'task_goal'})
    assert 'task_goal' not in json.loads(json_output_with_args)

def test_prepare_values_before_serialization(plan_step_2b: Plan):
    # Test case: Check if the method prepares values correctly before serialization.
    # This depends on what prepare_values_before_serialization actually does. 
    # Assuming it modifies some values in the object:
    original_state = plan_step_2b.dict()
    plan_step_2b.prepare_values_before_serialization()
    modified_state = plan_step_2b.dict()
    assert original_state != modified_state
    # Add more specific assertions based on expected behavior

def test_str_and_repr_methods(plan_step_2b: Plan):
    # Test case: Verify that __str__ and __repr__ return a string representation of the object.
    str_output = str(plan_step_2b)
    repr_output = repr(plan_step_2b)
    assert isinstance(str_output, str)
    assert isinstance(repr_output, str)
    # Optionally, add more detailed checks if you expect specific formats

def test_generate_uuid():
    # Test case: Check if it generates a unique UUID each time it's called.
    uuid1 = Task.generate_uuid()
    uuid2 = Task.generate_uuid()
    assert isinstance(uuid1, str) and isinstance(uuid2, str)
    assert uuid1 != uuid2
