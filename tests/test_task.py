from __future__ import annotations

# import copy
# import json
# import uuid

# import pytest
# from test_agents_planner import PLANNERAGENT

# from AFAAS.interfaces.task.meta import TaskStatusList
# from AFAAS.lib.task.plan import Plan
# from AFAAS.lib.task.task import Task

# plan_prepare_dinner = Plan(
#     task_id="100", task_goal="100. Prepare Dinner for Family", agent=PLANNERAGENT
# )
# PLANNERAGENT.plan = plan_prepare_dinner
# plan_prepare_dinner.state = TaskStatusList.READY

# task_101_buy_groceries = Task(task_id="101", task_goal="101. Buy Groceries")
# task_101_buy_groceries.state = TaskStatusList.READY

# task_102_clean_kitchen = Task(task_id="102", task_goal="102. Clean Kitchen")
# task_102_clean_kitchen.state = TaskStatusList.READY

# task_103_choose_music = Task(task_id="103", task_goal="103. Choose Dinner Music")
# task_103_choose_music.state = TaskStatusList.READY

# task_104_decorate_dining_room = Task(
#     task_id="104", task_goal="104. Decorate Dining Room"
# )
# task_104_decorate_dining_room.state = TaskStatusList.READY

# task_105_set_mood = Task(task_id="105", task_goal="105. Set the Mood for Dinner")
# task_105_set_mood.add_predecessor(task_103_choose_music)
# task_105_set_mood.add_predecessor(task_104_decorate_dining_room)

# task_106_set_table = Task(task_id="106", task_goal="106. Set the Table")
# task_106_set_table.add_predecessor(task_105_set_mood)

# task_107_make_salad = Task(task_id="107", task_goal="107. Make Salad")
# task_107_make_salad.add_predecessor(task_106_set_table)

# task_108_serve_salad = Task(task_id="108", task_goal="108. Serve Salad")
# task_108_serve_salad.add_predecessor(task_107_make_salad)

# task_200_prepare_main_course = Task(task_id="200", task_goal="200. Prepare Main Course")
# task_200_prepare_main_course.add_predecessor(task_106_set_table)

# task_201_serve_dinner = Task(task_id="201", task_goal="201. Serve Main Course")
# task_201_serve_dinner.add_predecessor(task_200_prepare_main_course)
# task_201_serve_dinner.add_predecessor(task_108_serve_salad)

# task_300_make_banana_bread = Task(task_id="300", task_goal="300. Make Banana Bread")

# task_300_1_gather_ingredients = Task(
#     task_id="300.1", task_goal="300.1. Gather Ingredients"
# )
# task_300_1_gather_ingredients.add_predecessor(task_101_buy_groceries)
# task_300_1_gather_ingredients.add_predecessor(task_102_clean_kitchen)

# task_300_2_prepare_baking_pan = Task(
#     task_id="300.2", task_goal="300.2. Prepare Baking Pan"
# )

# task_300_3_mix_ingredients = Task(task_id="300.3", task_goal="300.3. Mix Ingredients")
# task_300_3_mix_ingredients.add_predecessor(task_300_1_gather_ingredients)

# task_300_4_bake_bread = Task(task_id="300.4", task_goal="300.4. Bake the Bread")
# task_300_4_bake_bread.add_predecessor(task_201_serve_dinner)
# task_300_4_bake_bread.add_predecessor(task_300_3_mix_ingredients)

# task_300_5_cool_bread = Task(task_id="300.5", task_goal="300.5. Cool the Bread")
# task_300_5_cool_bread.add_predecessor(task_300_4_bake_bread)

# task_300_6_serve_bread = Task(task_id="300.6", task_goal="300.6. Serve Banana Bread")
# task_300_6_serve_bread.add_predecessor(task_300_5_cool_bread)
# task_300_6_serve_bread.add_predecessor(task_201_serve_dinner)

# # Subtasks for 'Mix Ingredients'
# task_300_3_1_measure_flour = Task(task_id="300.3.1", task_goal="300.3.1. Measure Flour")
# task_300_3_1_measure_flour.add_predecessor(task_300_1_gather_ingredients)

# task_300_3_2_mash_bananas = Task(task_id="300.3.2", task_goal="300.3.2. Mash Bananas")
# task_300_3_2_mash_bananas.add_predecessor(task_300_1_gather_ingredients)

# task_300_3_3_combine_wet_ingredients = Task(
#     task_id="300.3.3", task_goal="300.3.3. Combine Wet Ingredients"
# )
# task_300_3_3_combine_wet_ingredients.add_predecessor(task_300_3_1_measure_flour)
# task_300_3_3_combine_wet_ingredients.add_predecessor(task_300_3_2_mash_bananas)

# task_300_2_1_grease_pan = Task(
#     task_id="300.2.1", task_goal="300.2.1. Grease Baking Pan"
# )
# task_300_2_1_grease_pan.add_predecessor(task_300_1_gather_ingredients)

# task_300_2_2_line_pan = Task(
#     task_id="300.2.2", task_goal="300.2.2. Line Baking Pan with Parchment Paper"
# )
# task_300_2_2_line_pan.add_predecessor(task_300_2_1_grease_pan)

# task_300_1_1_find_ingredients_list = Task(
#     task_id="300.1.1", task_goal="300.1.1. Find Ingredients List"
# )
# task_300_1_1_find_ingredients_list.add_predecessor(task_101_buy_groceries)

# task_300_1_2_check_pantry = Task(
#     task_id="300.1.2", task_goal="300.1.2. Check Pantry for Ingredients"
# )
# task_300_1_2_check_pantry.add_predecessor(task_300_1_1_find_ingredients_list)

# # Adding tasks to plans
# plan_prepare_dinner.add_task(task_101_buy_groceries)
# plan_prepare_dinner.add_task(task_102_clean_kitchen)
# plan_prepare_dinner.add_task(task_103_choose_music)
# plan_prepare_dinner.add_task(task_104_decorate_dining_room)
# plan_prepare_dinner.add_task(task_105_set_mood)
# plan_prepare_dinner.add_task(task_106_set_table)
# plan_prepare_dinner.add_task(task_107_make_salad)
# plan_prepare_dinner.add_task(task_108_serve_salad)
# plan_prepare_dinner.add_task(task_200_prepare_main_course)
# plan_prepare_dinner.add_task(task_201_serve_dinner)
# plan_prepare_dinner.add_task(task_300_make_banana_bread)
# task_200_prepare_main_course.add_task(task_300_1_gather_ingredients)
# task_300_1_gather_ingredients.add_task(task_300_1_1_find_ingredients_list)
# task_300_1_gather_ingredients.add_task(task_300_1_2_check_pantry)
# task_200_prepare_main_course.add_task(task_300_2_prepare_baking_pan)
# task_300_2_prepare_baking_pan.add_task(task_300_2_1_grease_pan)
# task_300_2_prepare_baking_pan.add_task(task_300_2_2_line_pan)
# task_200_prepare_main_course.add_task(task_300_3_mix_ingredients)
# task_300_3_mix_ingredients.add_task(task_300_3_1_measure_flour)
# task_300_3_mix_ingredients.add_task(task_300_3_2_mash_bananas)
# task_300_3_mix_ingredients.add_task(task_300_3_3_combine_wet_ingredients)
# task_200_prepare_main_course.add_task(task_300_4_bake_bread)
# task_200_prepare_main_course.add_task(task_300_5_cool_bread)
# task_200_prepare_main_course.add_task(task_300_6_serve_bread)


# @pytest.fixture
# def plan_step_0():
#     # Initial setup with multiple subtasks
#     return copy.deepcopy(plan_prepare_dinner)


# @pytest.fixture
# def plan_step_1():
#     t: Plan = plan_step_0()
#     # Marking initial tasks as done
#     task_id: str
#     t._all_task_ids.reverse()

#     for task_id in t._all_task_ids:
#         task = t.get_task(task_id=task_id)
#         if task.state == TaskStatusList.READY and task.is_ready():
#             task.state = TaskStatusList.DONE
#         elif task.is_ready():
#             task.state = TaskStatusList.READY

#     t._all_task_ids.reverse()

#     for task_id in t._all_task_ids:
#         task = t.get_task(task_id=task_id)
#         print(f"Task {task.task_goal} is {task.state}")
#     return t


# @pytest.fixture
# def plan_step_2():
#     t: Plan = plan_step_1()
#     # Marking initial tasks as done
#     task_id: str
#     t._all_task_ids.reverse()

#     for task_id in t._all_task_ids:
#         task = t.get_task(task_id=task_id)
#         if task.state == TaskStatusList.READY and task.is_ready():
#             task.state = TaskStatusList.DONE
#         elif task.is_ready():
#             task.state = TaskStatusList.READY

#     t._all_task_ids.reverse()

#     for task_id in t._all_task_ids:
#         task = t.get_task(task_id=task_id)
#         print(f"Task {task.task_goal} is {task.state}")
#     return t


# # plan_step_1()
# plan_step_2()


# """
# @pytest.fixture
# def plan_step_2a():
#     t = plan_step_1()
#     t.get_task(task_id="45").state = TaskStatusList.DONE
#     t.get_task(task_id="300.1.1").state = TaskStatusList.DONE
#     t.get_task(task_id="106").state = TaskStatusList.DONE
#     return t


# @pytest.fixture
# def plan_step_2b():
#     t = plan_step_2a()
#     t.get_task(task_id="300.1.2").state = TaskStatusList.READY
#     t.get_task(task_id="107").state = TaskStatusList.READY
#     t.get_task(task_id="200").state = TaskStatusList.READY
#     t.get_task(task_id="200.1").state = TaskStatusList.READY
#     t.get_task(task_id="200.3").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_2c():
#     t = plan_step_2b()
#     t.get_task(task_id="300.1.2").state = TaskStatusList.DONE
#     # Task 300.1 can now be marked as done, as its subtasks are done
#     t.get_task(task_id="300.1").state = TaskStatusList.DONE
#     # Task 300.2 is not yet ready since its subtasks are not done
#     t.get_task(task_id="300.2").state = TaskStatusList.READY
#     t.get_task(task_id="300.2.1").state = TaskStatusList.READY
#     t.get_task(task_id="300.3").state = TaskStatusList.READY
#     t.get_task(task_id="300.3.1").state = TaskStatusList.READY
#     t.get_task(task_id="300.3.2").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_2d():
#     t = plan_step_2c()
#     t.get_task(task_id="107").state = TaskStatusList.DONE
#     t.get_task(task_id="200.1").state = TaskStatusList.DONE
#     t.get_task(task_id="200.2").state = TaskStatusList.READY
#     t.get_task(task_id="200.3").state = TaskStatusList.DONE
#     t.get_task(task_id="300.2.1").state = TaskStatusList.DONE
#     t.get_task(task_id="300.2.2").state = TaskStatusList.READY
#     t.get_task(task_id="200.2").state = TaskStatusList.DONE
#     t.get_task(task_id="200.4").state = TaskStatusList.READY
#     t.get_task(task_id="300.3.1").state = TaskStatusList.DONE
#     t.get_task(task_id="300.3.2").state = TaskStatusList.DONE
#     t.get_task(task_id="300.3.3").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_3():
#     t = plan_step_2b()

#     # Marking tasks as done, enabling other tasks to become read
#     t.get_task(task_id="200.4").state = TaskStatusList.DONE
#     t.get_task(task_id="200").state = TaskStatusList.DONE
#     # Task 300.2 can now be marked as ready
#     t.get_task(task_id="300.2.2").state = TaskStatusList.DONE
#     t.get_task(task_id="300.2").state = TaskStatusList.DONE

#     return t

# @pytest.fixture
# def plan_step_4():
#     t = plan_step_3()
#     t.get_task(task_id="300.3.3").state = TaskStatusList.DONE
#     t.get_task(task_id="300.3").state = TaskStatusList.DONE
#     t.get_task(task_id="300.4").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_5():
#     t = plan_step_4()
#     t.get_task(task_id="300.3").state = TaskStatusList.DONE
#     return t

# @pytest.fixture
# def plan_step_6():
#     t = plan_step_5()
#     t.get_task(task_id="108").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_7():
#     t = plan_step_6()
#     t.get_task(task_id="108").state = TaskStatusList.DONE
#     t.get_task(task_id="201").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_8():
#     t = plan_step_7()
#     t.get_task(task_id="201").state = TaskStatusList.DONE
#     return t

# @pytest.fixture
# def plan_step_9():
#     t = plan_step_8()
#     t.get_task(task_id="300.4").state = TaskStatusList.DONE
#     t.get_task(task_id="300.5").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_10():
#     t = plan_step_9()
#     t.get_task(task_id="300.5").state = TaskStatusList.DONE
#     t.get_task(task_id="300.6").state = TaskStatusList.READY
#     return t

# @pytest.fixture
# def plan_step_11():
#     t = plan_step_10()
#     t.get_task(task_id="300.6").state = TaskStatusList.DONE
#     return t


# @pytest.fixture
# def task_ready_no_predecessors_or_subtasks(plan_step_0: Plan):
#     # Task 'task_buy_groceries' has no predecessors or subtasks and is ready
#     return plan_step_0.get_task(task_id="5")

# @pytest.fixture
# def task_ready_all_predecessors_done(plan_step_2b: Plan):
#     # Task 'task_set_mood' with all predecessors done
#     return plan_step_2b.get_task(task_id="45")

# @pytest.fixture
# def task_ready_all_subtasks_done(plan_step_8: Plan):
#     # Task 'plan_prepare_dinner' with all subtasks done
#     return plan_step_8.get_task(task_id="1000")

# @pytest.fixture
# def task_with_mixed_predecessors(plan_step_7: Plan):
#     # Task 'task_serve_bread' with some predecessors done and some not
#     return plan_step_7.get_task(task_id="300.6")

# @pytest.mark.parametrize("plan_step, expected_task_ids", [
#     # Initial state, where tasks 5, 15, 25, 35, and 300.1.1 are ready and have no subtasks
#     (plan_step_0, ["5", "15", "25", "35", "300.1.1"]),


#     (plan_step_1, ["45", "106", "300.1.2"]),

#     # After subtask 300.1.2 is done, task 300.1 can be marked as done, and tasks 107, 200, 200.1 become ready
#     (plan_step_2a, ["45", "106", "107", "200", "200.1"]),

#     # After tasks 106, 200.1 are done, tasks 200.2, 200.3, 300.2, 300.3.1, and 300.3.2 become ready
#     (plan_step_3, ["45", "107", "200", "200.2", "200.3", "300.2", "300.3.1", "300.3.2"]),

#     # After tasks 200.2, 200.3, 300.3.1, 300.3.2 are done, tasks 200.4, 300.3.3, and 300.3 become ready
#     (plan_step_4, ["45", "107", "200", "200.4", "300.3.3", "300.3"]),

#     # After tasks 200.4, 300.3.3, 107, 200 are done, tasks 108, 300.4 become ready
#     (plan_step_5, ["45", "108", "300.3", "300.4"]),

#     # After tasks 108, 300.3 are done, tasks 201, 300.5 become ready
#     (plan_step_6, ["45", "201", "300.4", "300.5"]),

#     # After tasks 201, 300.4 are done, task 300.6 becomes ready
#     (plan_step_7, ["45", "300.5", "300.6"]),

#     # After task 300.5 is done, task 300.6 remains ready
#     (plan_step_8, ["45", "300.6"]),

#     # After task 300.6 is done, no more tasks are ready
#     (plan_step_9, ["45"]),

#     # After task 45 is done, no more tasks are ready
#     (plan_step_10, []),
# ])
# def test_get_ready_leaf_tasks(plan_step, expected_task_ids):
#     ready_tasks = plan_step.get_ready_leaf_tasks()
#     ready_task_ids = [task.task_id for task in ready_tasks]
#     assert set(ready_task_ids) == set(expected_task_ids)

# @pytest.mark.parametrize("plan_step, expected_task_id", [
#     (plan_step_0, "5"),   # "5. Buy Groceries" is ready
#     (plan_step_1, "45"),  # "45. Set the Mood for Dinner" is ready
#     (plan_step_2a, "106"), # "106. Set the Table" is ready
#     (plan_step_3, "300.1"),
#     (plan_step_4, "200.4"),
#     (plan_step_5, "108"), # "108. Serve Salad" is ready
#     (plan_step_6, "201"), # "201. Serve Main Course" is ready
#     (plan_step_7, "300.5"), # "300.5. Cool the Bread" is ready
#     (plan_step_8, "300.6"), # "300.6. Serve Banana Bread" is ready
#     (plan_step_9, "45"),  # "45. Set the Mood for Dinner" is ready
#     (plan_step_10, None)  # No task is ready
# ])
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
#     assert isinstance(encoded_data['test_field'], str)  # Assuming uuid is encoded to str

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
#     json_output_with_args = plan_step_2b.json(exclude={'task_goal'})
#     assert 'task_goal' not in json.loads(json_output_with_args)

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

# def test_generate_uuid():
#     # Test case: Check if it generates a unique UUID each time it's called.
#     uuid1 = Task.generate_uuid()
#     uuid2 = Task.generate_uuid()
#     assert isinstance(uuid1, str) and isinstance(uuid2, str)
#     assert uuid1 != uuid2
# """
