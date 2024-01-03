from __future__ import annotations

import copy
import json
import uuid

import pytest

from tests.dataset.agent_planner import agent_dataset

from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import Task

def plan_familly_dinner():
    agent = agent_dataset()
    plan_dict = {
    "created_at": "2023-12-31T15:38:45.666346",
    "modified_at": "2023-12-31T15:38:45.666355",
    "task_context": None,
    "long_description": None,
    "task_history": None,
    "acceptance_criteria": [],
    "tasks": [],
    "agent_id": agent.agent_id,
    "_task_predecessors": [],
    "_task_successors": [],
    "_subtasks": [
    ],
    "_modified_tasks_ids": [],
    "_new_tasks_ids": []
    }

    plan_prepare_dinner = Plan(
        plan_id = "pytest_P" + str(uuid.uuid4()) , task_goal="100. Prepare Dinner for Family", agent=agent, **plan_dict
    )
    agent.plan = plan_prepare_dinner
    #plan_prepare_dinner.state = TaskStatusList.READY

    task_101_buy_groceries = Task(agent = agent, task_id="101", task_goal="101. Buy Groceries")
    task_101_buy_groceries.state = TaskStatusList.READY

    task_102_clean_kitchen = Task(agent = agent, task_id="102", task_goal="102. Clean Kitchen")
    task_102_clean_kitchen.state = TaskStatusList.READY

    task_103_choose_music = Task(agent = agent, task_id="103", task_goal="103. Choose Dinner Music")
    task_103_choose_music.state = TaskStatusList.READY

    task_104_decorate_dining_room = Task(agent = agent, 
        task_id="104", task_goal="104. Decorate Dining Room"
    )
    task_104_decorate_dining_room.state = TaskStatusList.READY

    task_105_set_mood = Task(agent = agent, task_id="105", task_goal="105. Set the Mood for Dinner")
    task_105_set_mood.add_predecessor(task_103_choose_music)
    task_105_set_mood.add_predecessor(task_104_decorate_dining_room)

    task_106_set_table = Task(agent = agent, task_id="106", task_goal="106. Set the Table")
    task_106_set_table.add_predecessor(task_105_set_mood)

    task_107_make_salad = Task(agent = agent, task_id="107", task_goal="107. Make Salad")
    task_107_make_salad.add_predecessor(task_106_set_table)

    task_108_serve_salad = Task(agent = agent, task_id="108", task_goal="108. Serve Salad")
    task_108_serve_salad.add_predecessor(task_107_make_salad)

    task_200_prepare_main_course = Task(agent = agent, task_id="200", task_goal="200. Prepare Main Course")
    task_200_prepare_main_course.add_predecessor(task_106_set_table)

    task_201_serve_dinner = Task(agent = agent, task_id="201", task_goal="201. Serve Main Course")
    task_201_serve_dinner.add_predecessor(task_200_prepare_main_course)
    task_201_serve_dinner.add_predecessor(task_108_serve_salad)

    task_300_make_banana_bread = Task(agent = agent, task_id="300", task_goal="300. Make Banana Bread")

    task_300_1_gather_ingredients = Task(agent = agent, 
        task_id="300.1", task_goal="300.1. Gather Ingredients"
    )
    task_300_1_gather_ingredients.add_predecessor(task_101_buy_groceries)
    task_300_1_gather_ingredients.add_predecessor(task_102_clean_kitchen)

    task_300_2_prepare_baking_pan = Task(agent = agent, 
        task_id="300.2", task_goal="300.2. Prepare Baking Pan"
    )

    task_300_3_mix_ingredients = Task(agent = agent, task_id="300.3", task_goal="300.3. Mix Ingredients")
    task_300_3_mix_ingredients.add_predecessor(task_300_1_gather_ingredients)

    task_300_4_bake_bread = Task(agent = agent, task_id="300.4", task_goal="300.4. Bake the Bread")
    task_300_4_bake_bread.add_predecessor(task_201_serve_dinner)
    task_300_4_bake_bread.add_predecessor(task_300_3_mix_ingredients)

    task_300_5_cool_bread = Task(agent = agent, task_id="300.5", task_goal="300.5. Cool the Bread")
    task_300_5_cool_bread.add_predecessor(task_300_4_bake_bread)

    task_300_6_serve_bread = Task(agent = agent, task_id="300.6", task_goal="300.6. Serve Banana Bread")
    task_300_6_serve_bread.add_predecessor(task_300_5_cool_bread)
    task_300_6_serve_bread.add_predecessor(task_201_serve_dinner)

    # Subtasks for 'Mix Ingredients'
    task_300_3_1_measure_flour = Task(agent = agent, task_id="300.3.1", task_goal="300.3.1. Measure Flour")
    task_300_3_1_measure_flour.add_predecessor(task_300_1_gather_ingredients)

    task_300_3_2_mash_bananas = Task(agent = agent, task_id="300.3.2", task_goal="300.3.2. Mash Bananas")
    task_300_3_2_mash_bananas.add_predecessor(task_300_1_gather_ingredients)

    task_300_3_3_combine_wet_ingredients = Task(agent = agent, 
        task_id="300.3.3", task_goal="300.3.3. Combine Wet Ingredients"
    )
    task_300_3_3_combine_wet_ingredients.add_predecessor(task_300_3_1_measure_flour)
    task_300_3_3_combine_wet_ingredients.add_predecessor(task_300_3_2_mash_bananas)


    task_300_2_1_grease_pan = Task(agent = agent, 
        task_id="300.2.1", task_goal="300.2.1. Grease Baking Pan"
    )
    task_300_2_1_grease_pan.add_predecessor(task_300_1_gather_ingredients)

    task_300_2_2_line_pan = Task(agent = agent, 
        task_id="300.2.2", task_goal="300.2.2. Line Baking Pan with Parchment Paper"
    )
    task_300_2_2_line_pan.add_predecessor(task_300_2_1_grease_pan)


    task_300_1_1_find_ingredients_list = Task(agent = agent, 
        task_id="300.1.1", task_goal="300.1.1. Find Ingredients List"
    )
    task_300_1_1_find_ingredients_list.add_predecessor(task_101_buy_groceries)


    task_300_1_2_check_pantry = Task(agent = agent, 
        task_id="300.1.2", task_goal="300.1.2. Check Pantry for Ingredients"
    )
    task_300_1_2_check_pantry.add_predecessor(task_300_1_1_find_ingredients_list)

    plan_prepare_dinner.add_task(task_101_buy_groceries)

    plan_prepare_dinner.add_task(task_102_clean_kitchen)
    plan_prepare_dinner.add_task(task_103_choose_music)
    plan_prepare_dinner.add_task(task_104_decorate_dining_room)
    plan_prepare_dinner.add_task(task_105_set_mood)
    plan_prepare_dinner.add_task(task_106_set_table)
    plan_prepare_dinner.add_task(task_107_make_salad)
    plan_prepare_dinner.add_task(task_108_serve_salad)
    plan_prepare_dinner.add_task(task_200_prepare_main_course)
    plan_prepare_dinner.add_task(task_201_serve_dinner)
    plan_prepare_dinner.add_task(task_300_make_banana_bread)
    task_200_prepare_main_course.add_task(task_300_1_gather_ingredients)
    task_300_1_gather_ingredients.add_task(task_300_1_1_find_ingredients_list)
    task_300_1_gather_ingredients.add_task(task_300_1_2_check_pantry)
    task_200_prepare_main_course.add_task(task_300_2_prepare_baking_pan)
    task_300_2_prepare_baking_pan.add_task(task_300_2_1_grease_pan)
    task_300_2_prepare_baking_pan.add_task(task_300_2_2_line_pan)
    task_200_prepare_main_course.add_task(task_300_3_mix_ingredients)
    task_300_3_mix_ingredients.add_task(task_300_3_1_measure_flour)
    task_300_3_mix_ingredients.add_task(task_300_3_2_mash_bananas)
    task_300_3_mix_ingredients.add_task(task_300_3_3_combine_wet_ingredients)
    task_200_prepare_main_course.add_task(task_300_4_bake_bread)
    task_200_prepare_main_course.add_task(task_300_5_cool_bread)
    task_200_prepare_main_course.add_task(task_300_6_serve_bread)

    return plan_prepare_dinner



@pytest.fixture
def plan_step_0():
    # Initial setup with multiple subtasks
    return plan_familly_dinner()


@pytest.fixture(scope='function')
def plan_step_1()-> Plan:
    t = plan_step_0()
    # Marking initial tasks as done
    task_id: str
    t._all_task_ids.reverse()

    for task_id in t._all_task_ids:
        task = t.get_task(task_id=task_id)
        if task.state == TaskStatusList.READY and task.is_ready():
            task.state = TaskStatusList.DONE
            task.task_text_output = f"Completed '{task.task_goal}' successfully."
        elif task.is_ready():
            task.state = TaskStatusList.READY

    t._all_task_ids.reverse()

    for task_id in t._all_task_ids:
        task = t.get_task(task_id=task_id)
        print(f"Task {task.task_goal} is {task.state}")
    return copy.deepcopy(t)



# @pytest.fixture(scope='function')
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
#     return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_2a()-> Plan:
    t = plan_step_1()

    # Task 45 completed (Example task, adjust as needed)
    t.get_task(task_id="45").state = TaskStatusList.DONE
    t.get_task(task_id="45").task_text_output = "Completed task 45 successfully."

    # Task 300.1.1: Find Ingredients List
    t.get_task(task_id="300.1.1").state = TaskStatusList.DONE
    t.get_task(task_id="300.1.1").task_text_output = "Found the complete ingredients list for banana bread."

    # Task 106: Set the Table
    t.get_task(task_id="106").state = TaskStatusList.DONE
    t.get_task(task_id="106").task_text_output = "Set the table for dinner, including plates, cutlery, and glasses for all guests."

    return copy.deepcopy(t)



@pytest.fixture(scope='function')
def plan_step_2b()-> Plan:
    t = plan_step_2a()
    t.get_task(task_id="300.1.2").state = TaskStatusList.READY
    t.get_task(task_id="107").state = TaskStatusList.READY
    t.get_task(task_id="200").state = TaskStatusList.READY
    t.get_task(task_id="200.1").state = TaskStatusList.READY
    t.get_task(task_id="200.3").state = TaskStatusList.READY
    return copy.deepcopy(t)

@pytest.fixture(scope='function')
def plan_step_2c()-> Plan:
    t = plan_step_2b()
    t.get_task(task_id="300.1.2").state = TaskStatusList.DONE
    t.get_task(task_id="300.1.2").task_text_output = "Checked the pantry and gathered all necessary ingredients for the banana bread."

    # Task 300.1 can now be marked as done, as its subtasks are done
    t.get_task(task_id="300.1").state = TaskStatusList.DONE
    t.get_task(task_id="300.1").task_text_output = "Successfully gathered all ingredients required for making banana bread."

    # Task 300.2 is not yet ready since its subtasks are not done
    t.get_task(task_id="300.2").state = TaskStatusList.READY
    t.get_task(task_id="300.2.1").state = TaskStatusList.READY
    t.get_task(task_id="300.3").state = TaskStatusList.READY
    t.get_task(task_id="300.3.1").state = TaskStatusList.READY
    t.get_task(task_id="300.3.2").state = TaskStatusList.READY
    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_2d()-> Plan:
    t = plan_step_2c()
    t.get_task(task_id="107").state = TaskStatusList.DONE
    t.get_task(task_id="107").task_text_output = "Prepared a fresh and healthy salad with a variety of greens and dressing."

    t.get_task(task_id="200.1").state = TaskStatusList.DONE
    t.get_task(task_id="200.1").task_text_output = "Completed the initial preparations for the main course."

    t.get_task(task_id="200.3").state = TaskStatusList.DONE
    t.get_task(task_id="200.3").task_text_output = "Finalized cooking the main course, ensuring it's delicious and well-seasoned."

    t.get_task(task_id="300.2.1").state = TaskStatusList.DONE
    t.get_task(task_id="300.2.1").task_text_output = "Greased the baking pan for the banana bread, making it ready for the batter."

    t.get_task(task_id="300.3.1").state = TaskStatusList.DONE
    t.get_task(task_id="300.3.1").task_text_output = "Measured the required amount of flour for the banana bread accurately."

    t.get_task(task_id="300.3.2").state = TaskStatusList.DONE
    t.get_task(task_id="300.3.2").task_text_output = "Mashed bananas to the right consistency for the banana bread mixture."

    t.get_task(task_id="300.3.3").state = TaskStatusList.READY
    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_3()-> Plan:
    t = plan_step_2b()

    # Completing task 200.4
    t.get_task(task_id="200.4").state = TaskStatusList.DONE
    t.get_task(task_id="200.4").task_text_output = "Successfully finished the intricate preparation of the dessert, carefully arranging and garnishing each component to create a visually stunning and delicious finale to the meal."

    # Marking the main course task as done
    t.get_task(task_id="200").state = TaskStatusList.DONE
    t.get_task(task_id="200").task_text_output = "The main course has been masterfully prepared, combining a balance of flavors and textures, ensuring each ingredient complements the others to create a harmonious and satisfying dish."

    # Completing task 300.2.2 and marking 300.2 as done
    t.get_task(task_id="300.2.2").state = TaskStatusList.DONE
    t.get_task(task_id="300.2.2").task_text_output = "The baking pan has been lined with parchment paper, ensuring a non-stick surface which will aid in the seamless removal of the banana bread once baked, contributing to a perfect presentation."

    t.get_task(task_id="300.2").state = TaskStatusList.DONE
    t.get_task(task_id="300.2").task_text_output = "All preparatory steps for baking the banana bread are complete. The pan is well-prepared and the oven preheated, setting the stage for a delicious and aromatic baking experience."

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_4()-> Plan:
    t = plan_step_3()

    # Completing the combination of wet ingredients for the banana bread
    t.get_task(task_id="300.3.3").state = TaskStatusList.DONE
    t.get_task(task_id="300.3.3").task_text_output = "Successfully combined the wet ingredients, including the mashed bananas, eggs, and butter, ensuring a smooth and well-integrated mixture that will contribute to the moist and rich texture of the banana bread."

    # Marking the mixing of ingredients as done
    t.get_task(task_id="300.3").state = TaskStatusList.DONE
    t.get_task(task_id="300.3").task_text_output = "All ingredients for the banana bread have been meticulously mixed, achieving a perfect balance of flavors. The consistency of the batter is ideal, promising a delightful texture and taste upon baking."

    # Preparing to bake the bread
    t.get_task(task_id="300.4").state = TaskStatusList.READY

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_5()-> Plan:
    t = plan_step_4()

    # Reiterating completion of the mixing of ingredients
    t.get_task(task_id="300.3").state = TaskStatusList.DONE
    t.get_task(task_id="300.3").task_text_output = "The final mix of the banana bread ingredients is ready, with each component blended harmoniously. The aroma of the batter hints at the delicious baked good to come, setting high expectations for the finished product."

    return copy.deepcopy(t)

@pytest.fixture(scope='function')
def plan_step_6()-> Plan:
    t = plan_step_5()

    # Making salad serving task ready
    t.get_task(task_id="108").state = TaskStatusList.READY

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_7()-> Plan:
    t = plan_step_6()

    # Completing the salad serving task
    t.get_task(task_id="108").state = TaskStatusList.DONE
    t.get_task(task_id="108").task_text_output = "The salad, a symphony of fresh greens and vibrant vegetables, was elegantly served in a large, ornate bowl. Each portion was carefully dressed with a homemade vinaigrette, ensuring a perfect blend of acidity and sweetness that complemented the fresh produce. The presentation was as delightful as the taste, with a sprinkle of herbs and edible flowers adding a touch of sophistication and color."

    # Setting the main course serving task as ready
    t.get_task(task_id="201").state = TaskStatusList.READY

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_8()-> Plan:
    t = plan_step_7()

    # Completing the main course serving task
    t.get_task(task_id="201").state = TaskStatusList.DONE
    t.get_task(task_id="201").task_text_output = "The main course, a masterpiece of culinary expertise, was served with precision and grace. Each plate was a canvas showcasing the chef's skill, with the main dish taking center stage, surrounded by a medley of side dishes that enhanced its flavors. The aroma wafting from the plates promised a savory experience, and the first bite did not disappoint, offering a burst of flavors that danced on the palate. The combination of textures, colors, and tastes made for a memorable dining experience, leaving the guests eagerly anticipating the next course."

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_9()-> Plan:
    t = plan_step_8()

    # Completing the baking of banana bread
    t.get_task(task_id="300.4").state = TaskStatusList.DONE
    t.get_task(task_id="300.4").task_text_output = "The aroma of freshly baked banana bread filled the air, a testament to the perfect blend of ripe bananas, sugar, and spices. As the bread cooled on the wire rack, its golden-brown crust and moist interior were a promise of the delightful taste to come. The kitchen was imbued with a sense of warmth and homeliness, with the banana bread's comforting aroma invoking fond memories and a sense of anticipation for the treat ahead."

    # Setting the task of cooling the banana bread as ready
    t.get_task(task_id="300.5").state = TaskStatusList.READY

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_10()-> Plan:
    t = plan_step_9()

    # Completing the cooling of banana bread
    t.get_task(task_id="300.5").state = TaskStatusList.DONE
    t.get_task(task_id="300.5").task_text_output = "The banana bread, having cooled to the perfect temperature, sat majestically on its serving platter. The cooling process had allowed the flavors to meld together beautifully, enhancing the bread's rich, sweet taste. The crust had settled into a delightful texture, slightly crisp on the outside while preserving the bread's tender, moist crumb. It was now the ideal time to serve this homemade delicacy, a sweet ending to an exquisite meal."

    # Setting the task of serving the banana bread as ready
    t.get_task(task_id="300.6").state = TaskStatusList.READY

    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def plan_step_11()-> Plan:
    t = plan_step_10()

    # Completing the serving of banana bread
    t.get_task(task_id="300.6").state = TaskStatusList.DONE
    t.get_task(task_id="300.6").task_text_output = "The moment of serving the banana bread was a culmination of anticipation and delight. Each slice was carefully cut, revealing the moist, tender texture interspersed with the subtle sweetness of bananas and a hint of cinnamon. As the bread was served, the guests expressed their admiration for its perfect texture and exquisite flavor. The banana bread was not just a dessert; it was a celebration of homemade baking, bringing a sweet and satisfying conclusion to the family dinner."

    return copy.deepcopy(t)



@pytest.fixture
def task_ready_no_predecessors_or_subtasks(plan_step_0: Plan) -> Task:
    # Task 'task_101_buy_groceries' has no predecessors or subtasks and is ready
    t = plan_step_0.get_task(task_id="101")
    return t



@pytest.fixture(scope='function')
def task_ready_all_predecessors_done(plan_step_2b: Plan) -> Task:
    # Task 'task_300_2_2_line_pan' with all predecessors done
    t = plan_step_2b.get_task(task_id="300.2.2")
    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def task_ready_all_subtasks_done(plan_step_8: Plan) -> Task:
    # Task 'task_300_1_gather_ingredients' with all subtasks done
    t = plan_step_8.get_task(task_id="300.1")
    return copy.deepcopy(t)

@pytest.fixture(scope='function')
def task_with_mixed_predecessors(plan_step_7: Plan) -> Task:
    # Task 'task_300_6_serve_bread' with some predecessors done and some not
    t = plan_step_7.get_task(task_id="300.6")
    return copy.deepcopy(t)

@pytest.fixture(scope='function')
def task_with_unmet_predecessors(plan_step_0: Plan) -> Task:
    # Task '300.3.3' has unmet predecessors at plan_step_0
    t = plan_step_0.get_task(task_id="300.3.3")
    return copy.deepcopy(t)

@pytest.fixture(scope='function')
def task_with_ongoing_subtasks(plan_step_0: Plan) -> Task:
    # Task '200' has ongoing subtasks at plan_step_0
    t = plan_step_0.get_task(task_id="200")
    return copy.deepcopy(t)


@pytest.fixture(scope='function')
def task_awaiting_preparation(plan_step_0: Plan) -> Task:
    # Task '300.4' is not ready yet at plan_step_0
    t = plan_step_0.get_task(task_id="300.4")
    return copy.deepcopy(t)

