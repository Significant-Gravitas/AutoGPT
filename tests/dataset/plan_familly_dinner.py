from __future__ import annotations

import copy
import json
import uuid

import pytest

from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import Task
from tests.dataset.agent_planner import agent_dataset
from tests.utils.ascii_tree import make_tree


async def _plan_familly_dinner():
    agent = await agent_dataset()
    plan_dict = {
        "created_at": "2023-12-31T15:38:45.666346",
        "modified_at": "2023-12-31T15:38:45.666355",
        "task_context": None,
        "task_history": None,
        "acceptance_criteria": [],
        "tasks": [],
        "agent_id": agent.agent_id,
        "_task_predecessors": [],
        "_task_successors": [],
        "_subtasks": [],
        "_modified_tasks_ids": [],
        "_new_tasks_ids": [],
    }

    plan_prepare_dinner = Plan(
        plan_id="pytest_P" + str(uuid.uuid4()),
        task_goal="100. Prepare Dinner for Family",
        long_description="Coordinate and execute all necessary activities to prepare a dinner for the family, including grocery shopping, kitchen preparation, cooking, and setting the mood.",
        agent=agent,
        **plan_dict,
    )
    agent.plan = plan_prepare_dinner
    plan_prepare_dinner._subtasks = None
    plan_prepare_dinner._all_task_ids = []
    plan_prepare_dinner._ready_task_ids = []
    plan_prepare_dinner._done_task_ids = []
    plan_prepare_dinner._loaded_tasks_dict = {}
    plan_prepare_dinner._modified_tasks_ids = []
    plan_prepare_dinner._new_tasks_ids = []
    return plan_prepare_dinner


async def plan_familly_dinner_with_tasks():
    plan_prepare_dinner = await _plan_familly_dinner()
    agent = plan_prepare_dinner.agent
    task_101_buy_groceries = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="101",
        task_goal="101. Buy Groceries",
        long_description="Procure all necessary ingredients and items from a grocery store required for preparing the dinner.",
    )
    task_102_clean_kitchen = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="102",
        task_goal="102. Clean Kitchen",
        long_description="Thoroughly clean and organize the kitchen area to create a hygienic and efficient cooking environment.",
    )
    task_103_choose_music = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="103",
        task_goal="103. Choose Dinner Music",
        long_description="Select and arrange a playlist of music that will create a pleasant and relaxing atmosphere during the dinner.",
    )
    task_104_decorate_dining_room = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="104",
        task_goal="104. Decorate Dining Room",
        long_description="Enhance the dining room's ambiance by arranging decorations, setting appropriate lighting, and ensuring a visually pleasing and comfortable dining environment.",
    )

    task_105_set_mood = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="105",
        task_goal="105. Set the Mood for Dinner",
        long_description="Create a welcoming and enjoyable dinner atmosphere by adjusting the lighting, music, and room temperature, and ensuring all elements contribute to a pleasant dining experience.",
    )
    task_101_buy_groceries.state = TaskStatusList.READY
    task_102_clean_kitchen.state = TaskStatusList.READY
    task_103_choose_music.state = TaskStatusList.READY
    task_104_decorate_dining_room.state = TaskStatusList.READY
    task_105_set_mood.add_predecessor(task_103_choose_music)
    task_105_set_mood.add_predecessor(task_104_decorate_dining_room)

    task_106_set_table = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="106",
        task_goal="106. Set the Table",
        long_description="Arrange the dining table with necessary cutlery, plates, glasses, and napkins, ensuring it's elegantly set and ready for the meal.",
    )
    task_106_set_table.add_predecessor(task_105_set_mood)

    task_107_make_salad = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="107",
        task_goal="107. Make Salad",
        long_description="Prepare a fresh salad by selecting and combining a variety of ingredients, dressing it appropriately, and presenting it in an appealing manner.",
    )
    task_107_make_salad.add_predecessor(task_106_set_table)

    task_108_serve_salad = Task(
        agent=agent,
        task_id="108",
        task_goal="108. Serve Salad",
        long_description="Present the prepared salad to the diners, ensuring it's served at the right temperature and in an appealing way, as a starter for the dinner.",
    )
    task_108_serve_salad.add_predecessor(task_107_make_salad)

    task_200_prepare_main_course = Task(
        agent=agent,
        task_id="200",
        task_goal="200. Prepare Main Course",
        long_description="Cook the main course of the dinner, focusing on flavor, presentation, and ensuring it meets the dietary preferences and needs of the family.",
    )
    task_200_prepare_main_course.add_predecessor(task_106_set_table)

    task_201_serve_dinner = Task(
        agent=agent,
        task_id="201",
        task_goal="201. Serve Main Course",
        long_description="Gracefully serve the prepared main course to the diners, ensuring each person receives their portion with appropriate accompaniments.",
    )
    task_201_serve_dinner.add_predecessor(task_200_prepare_main_course)
    task_201_serve_dinner.add_predecessor(task_108_serve_salad)

    task_300_make_banana_bread = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300",
        task_goal="300. Make Banana Bread",
        long_description="Bake a banana bread by following a specific recipe, focusing on achieving the right texture and flavor, and ensuring it's enjoyable for all diners.",
    )

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
    return plan_prepare_dinner


async def plan_familly_dinner_with_tasks_saved_in_db():
    plan_prepare_dinner = await plan_familly_dinner_with_tasks()
    await plan_prepare_dinner.db_create()
    await plan_prepare_dinner.db_save()
    return plan_prepare_dinner


async def move_to_next(plan: Plan, only_once=True):
    task_id: str
    # plan._all_task_ids.reverse()
    return_flag = False

    for task_id in plan._all_task_ids:
        task = await plan.get_task(task_id=task_id)
        #
        # raise Exception(task.debug_formated_str() + await make_tree(plan))
        if (
            not return_flag
            and task.state == TaskStatusList.READY
            and await task.is_ready()
        ):
            await task.close_task()
            task.task_text_output = f"Completed '{task.task_goal}' successfully."
            if only_once:
                return_flag = True

        if task.state == TaskStatusList.BACKLOG:
            if await task.is_ready():
                task.state = TaskStatusList.READY
                if return_flag:
                    return plan

    # plan._all_task_ids.reverse()

    return plan


async def reorder_list(plan: Plan, task=Task):
    lst = plan._all_task_ids
    item = task.task_id
    num_last_items = len(task.subtasks.get_all_task_ids_from_stack())
    # Check if item is in the list and num_last_items is valid
    if item not in lst or num_last_items > len(lst) or num_last_items < 0:
        raise ValueError(f"Invalid item or number of last items {num_last_items}")

    # Find the index of the specified item
    item_index = lst.index(item)

    # Rearrange the list
    # Extract the last 'num_last_items' elements
    last_elements = lst[-num_last_items:]

    # Remove these elements from their original position
    lst = lst[:-num_last_items]

    # Update the state of tasks after the specified item
    for x in lst[item_index + 1 :]:
        parent_sibling = await plan.get_task(x)
        parent_sibling.state = TaskStatusList.BACKLOG

    # Insert them after the specified item
    lst[item_index + 1 : item_index + 1] = last_elements

    plan._all_task_ids = lst


@pytest.fixture(scope="function")
async def plan_with_no_task():
    plan = await _plan_familly_dinner()
    yield plan


@pytest.fixture(scope="function")
async def plan_step_0() -> Plan:
    # Initial setup with multiple subtasks
    return await plan_familly_dinner_with_tasks_saved_in_db()


@pytest.fixture(scope="function")
async def plan_step_1():
    plan = await _plan_step_1()
    yield plan


async def _plan_step_1() -> Plan:
    t = await plan_familly_dinner_with_tasks_saved_in_db()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_2():
    plan = await _plan_step_2()
    yield plan


async def _plan_step_2() -> Plan:
    t = await _plan_step_1()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_3():
    plan = await _plan_step_3()
    yield plan


async def _plan_step_3() -> Plan:
    t = await _plan_step_2()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_4():
    plan = await _plan_step_4()
    yield plan


async def _plan_step_4() -> Plan:
    t = await _plan_step_3()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_5():
    plan = await _plan_step_5()
    yield plan


async def _plan_step_5() -> Plan:
    t = await _plan_step_4()
    plan = await move_to_next(t)

    return plan


@pytest.fixture(scope="function")
async def plan_step_6():
    plan = await _plan_step_6()
    yield plan


async def _plan_step_6() -> Plan:
    t = await _plan_step_5()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_7():
    plan = await _plan_step_7()
    yield plan


async def _plan_step_7() -> Plan:
    t = await _plan_step_6()
    plan = await move_to_next(t)
    # raise Exception(str(plan._all_task_ids) + "\n\n\n" +  await make_tree(plan))
    return plan


@pytest.fixture(scope="function")
async def plan_step_8():
    plan = await _plan_step_8()
    yield plan


async def _plan_step_8() -> Plan:
    t = await _plan_step_7()
    plan = await move_to_next(t)
    agent = plan.agent

    task_200_prepare_main_course = await plan.get_task("200")
    task_300_1_gather_ingredients = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.1",
        task_goal="300.1. Gather Ingredients",
        long_description="Collect all necessary ingredients for baking banana bread, ensuring they are fresh and of good quality.",
    )
    task_300_1_gather_ingredients.add_predecessor(await plan.get_task("101"))
    task_300_1_gather_ingredients.add_predecessor(await plan.get_task("102"))

    task_300_2_prepare_baking_pan = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.2",
        task_goal="300.2. Prepare Baking Pan",
        long_description="Ready the baking pan for the banana bread by properly greasing it or lining it with parchment paper, ensuring the bread will not stick and will bake evenly.",
    )
    task_300_3_mix_ingredients = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.3",
        task_goal="300.3. Mix Ingredients",
        long_description="Combine the banana bread ingredients in the correct sequence and method, ensuring a consistent and well-mixed batter.",
    )
    task_300_3_mix_ingredients.add_predecessor(task_300_1_gather_ingredients)

    task_300_4_bake_bread = Task(
        agent=agent,
        task_id="300.4",
        task_goal="300.4. Bake the Bread",
        long_description="Bake the banana bread in a preheated oven, monitoring it to achieve the perfect bake, both in terms of texture and color.",
    )
    task_300_4_bake_bread.add_predecessor(task_300_1_gather_ingredients)
    task_300_4_bake_bread.add_predecessor(task_300_3_mix_ingredients)

    task_300_5_cool_bread = Task(
        agent=agent,
        task_id="300.5",
        task_goal="300.5. Cool the Bread",
        long_description="Allow the baked banana bread to cool down to an appropriate temperature before serving, ensuring it retains its texture and flavor.",
    )
    task_300_5_cool_bread.add_predecessor(task_300_4_bake_bread)

    task_300_6_serve_bread = Task(
        agent=agent,
        task_id="300.6",
        task_goal="300.6. Serve Banana Bread",
        long_description="Serve the cooled banana bread, slicing it neatly and presenting it in an appetizing manner, potentially with accompaniments like butter or cream.",
    )
    task_300_6_serve_bread.add_predecessor(task_300_5_cool_bread)
    task_300_6_serve_bread.add_predecessor(task_300_1_gather_ingredients)
    # raise Exception(str(plan._all_task_ids) + "\n\n\n" +  await make_tree(plan))
    task_200_prepare_main_course.add_task(task_300_1_gather_ingredients)
    task_200_prepare_main_course.add_task(task_300_2_prepare_baking_pan)
    task_200_prepare_main_course.add_task(task_300_3_mix_ingredients)
    task_200_prepare_main_course.add_task(task_300_4_bake_bread)
    task_200_prepare_main_course.add_task(task_300_5_cool_bread)
    task_200_prepare_main_course.add_task(task_300_6_serve_bread)
    await reorder_list(plan, task_200_prepare_main_course)
    return plan


@pytest.fixture(scope="function")
async def plan_step_9():
    plan = await _plan_step_9()
    yield plan


async def _plan_step_9() -> Plan:
    t = await _plan_step_8()
    plan = await move_to_next(t)
    agent = plan.agent
    task_300_1_gather_ingredients = await plan.get_task("300.1")

    task_300_1_1_find_ingredients_list = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.1.1",
        task_goal="300.1.1. Find Ingredients List",
    )
    task_300_1_1_find_ingredients_list.add_predecessor(await plan.get_task("101"))

    task_300_1_2_check_pantry = Task(
        agent=agent,
        task_id="300.1.2",
        task_goal="300.1.2. Check Pantry for Ingredients",
    )
    # raise Exception(str(plan._all_task_ids) + "\n\n\n" +  await make_tree(plan))
    task_300_1_gather_ingredients.add_task(task_300_1_1_find_ingredients_list)
    task_300_1_gather_ingredients.add_task(task_300_1_2_check_pantry)
    task_300_1_2_check_pantry.add_predecessor(task_300_1_1_find_ingredients_list)
    await reorder_list(plan, task_300_1_gather_ingredients)
    return plan


@pytest.fixture(scope="function")
async def plan_step_10():
    plan = await _plan_step_10()
    yield plan


async def _plan_step_10() -> Plan:
    t = await _plan_step_9()
    plan = await move_to_next(t)
    agent = plan.agent

    task_300_2_1_grease_pan = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.2.1",
        task_goal="300.2.1. Grease Baking Pan",
    )
    task_300_2_1_grease_pan.add_predecessor(await plan.get_task("300.1"))

    task_300_2_2_line_pan = Task(
        agent=agent,
        task_id="300.2.2",
        task_goal="300.2.2. Line Baking Pan with Parchment Paper",
    )
    task_300_2_2_line_pan.add_predecessor(task_300_2_1_grease_pan)
    task_300_2_prepare_baking_pan = await plan.get_task("300.2")
    # raise Exception(str(plan._all_task_ids) + "\n\n\n" +  await make_tree(plan))
    task_300_2_prepare_baking_pan.add_task(task_300_2_1_grease_pan)
    task_300_2_prepare_baking_pan.add_task(task_300_2_2_line_pan)
    await reorder_list(plan, task_300_2_prepare_baking_pan)
    return plan


@pytest.fixture(scope="function")
async def plan_step_11():
    plan = await _plan_step_11()
    yield plan


async def _plan_step_11() -> Plan:
    t = await _plan_step_10()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_12():
    plan = await _plan_step_12()
    yield plan


async def _plan_step_12() -> Plan:
    t = await _plan_step_11()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_13():
    plan = await _plan_step_13()
    yield plan


async def _plan_step_13() -> Plan:
    t = await _plan_step_12()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_14():
    plan = await _plan_step_14()
    yield plan


async def _plan_step_14() -> Plan:
    t = await _plan_step_13()
    plan = await move_to_next(t)
    agent = plan.agent
    task_300_3_mix_ingredients = await plan.get_task("300.3")

    # Subtasks for 'Mix Ingredients'
    task_300_3_1_measure_flour = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.3.1",
        task_goal="300.3.1. Measure Flour",
    )
    task_300_3_1_measure_flour.add_predecessor(await plan.get_task("300.1"))

    task_300_3_2_mash_bananas = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.3.2",
        task_goal="300.3.2. Mash Bananas",
    )
    task_300_3_2_mash_bananas.add_predecessor(await plan.get_task("300.1"))

    task_300_3_3_combine_wet_ingredients = Task(
        agent=agent,
        plan_id=agent.plan.plan_id,
        task_id="300.3.3",
        task_goal="300.3.3. Combine Wet Ingredients",
    )
    task_300_3_3_combine_wet_ingredients.add_predecessor(task_300_3_1_measure_flour)
    task_300_3_3_combine_wet_ingredients.add_predecessor(task_300_3_2_mash_bananas)
    # raise Exception(str(plan._all_task_ids) + "\n\n\n" +  await make_tree(plan))
    task_300_3_mix_ingredients.add_task(task_300_3_1_measure_flour)
    task_300_3_mix_ingredients.add_task(task_300_3_2_mash_bananas)
    task_300_3_mix_ingredients.add_task(task_300_3_3_combine_wet_ingredients)
    await reorder_list(plan, task_300_3_mix_ingredients)
    # fix_201 = await plan.get_task("201")
    # if not await fix_201.is_ready() :

    #     raise Exception(
    #                      "plan : " + str(plan._all_task_ids) + "\n" ,
    #                      "predecessors : " + str( fix_201._task_predecessors.get_all_task_ids_from_stack()) + "\n" ,
    #                      await make_tree(plan)
    #                     )

    return plan


@pytest.fixture(scope="function")
async def plan_step_15():
    plan = await _plan_step_15()
    yield plan


async def _plan_step_15() -> Plan:
    t = await _plan_step_14()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_16():
    plan = await _plan_step_16()
    yield plan


async def _plan_step_16() -> Plan:
    t = await _plan_step_15()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_17():
    plan = await _plan_step_17()
    yield plan


async def _plan_step_17() -> Plan:
    t = await _plan_step_16()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_18():
    plan = await _plan_step_18()
    yield plan


async def _plan_step_18() -> Plan:
    t = await _plan_step_17()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_19():
    plan = await _plan_step_19()
    yield plan


async def _plan_step_19() -> Plan:
    t = await _plan_step_18()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_20():
    plan = await _plan_step_20()
    yield plan


async def _plan_step_20() -> Plan:
    t = await _plan_step_19()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_21():
    plan = await _plan_step_21()
    yield plan


async def _plan_step_21() -> Plan:
    t = await _plan_step_20()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_22():
    plan = await _plan_step_22()
    yield plan


async def _plan_step_22() -> Plan:
    t = await _plan_step_21()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_23():
    plan = await _plan_step_23()
    yield plan


async def _plan_step_23() -> Plan:
    t = await _plan_step_22()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_24():
    plan = await _plan_step_24()
    yield plan


async def _plan_step_24() -> Plan:
    t = await _plan_step_23()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_25():
    plan = await _plan_step_25()
    yield plan


async def _plan_step_25() -> Plan:
    t = await _plan_step_24()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_26():
    plan = await _plan_step_26()
    yield plan


async def _plan_step_26() -> Plan:
    t = await _plan_step_25()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_27():
    plan = await _plan_step_27()
    yield plan


async def _plan_step_27() -> Plan:
    t = await _plan_step_26()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_28():
    plan = await _plan_step_28()
    yield plan


async def _plan_step_28() -> Plan:
    t = await _plan_step_27()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_29():
    plan = await _plan_step_29()
    yield plan


async def _plan_step_29() -> Plan:
    t = await _plan_step_28()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_30():
    plan = await _plan_step_30()
    yield plan


async def _plan_step_30() -> Plan:
    t = await _plan_step_29()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_31():
    plan = await _plan_step_31()
    yield plan


async def _plan_step_31() -> Plan:
    t = await _plan_step_30()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_32():
    plan = await _plan_step_32()
    yield plan


async def _plan_step_32() -> Plan:
    t = await _plan_step_31()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def plan_step_33():
    plan = await _plan_step_33()
    yield plan


async def _plan_step_33() -> Plan:
    t = await _plan_step_32()
    plan = await move_to_next(t)
    return plan


@pytest.fixture(scope="function")
async def task_ready_no_predecessors_or_subtasks() -> Task:
    return await _default_task()


@pytest.fixture(scope="function")
async def default_task():
    return await _default_task()


async def _default_task():
    """
    Ensures task 'task_101_buy_groceries' has no active predecessors or subtasks.
    Raises an exception if conditions are not met.
    """
    plan = await plan_familly_dinner_with_tasks_saved_in_db()
    task = await plan.get_task(task_id="101")
    active_predecessors = await task._task_predecessors.get_active_tasks_from_stack()
    active_subtasks = await task.subtasks.get_active_tasks_from_stack()

    if active_predecessors or active_subtasks:
        error_message = (
            "Error: Fixture task_awaiting_no_active_predecessors_or_subtasks needs update\n"
            f"Active predecessors: {len(active_predecessors)}, active subtasks: {len(active_subtasks)}.\n\n\n"
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)
    if task.state != TaskStatusList.READY:
        raise Exception(
            f"Error: Fixture _default_task should be ready\n\n\n {await make_tree(plan)}"
        )
    await plan.db_save()
    return task


@pytest.fixture(scope="function")
async def task_with_unmet_predecessors(plan_step_14: Plan) -> Task:
    plan = plan_step_14  # Added line
    task = await plan.get_task(task_id="300.3.3")
    active_predecessors = await task._task_predecessors.get_active_tasks_from_stack()

    if not active_predecessors:
        error_message = (
            "Error: Fixture task_with_unmet_predecessors needs update. "
            f"There are {len(active_predecessors)} active predecessors."
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)

    if task.state != TaskStatusList.BACKLOG:
        error_message = (
            "Error: Fixture task_awaiting_preparation needs update. "
            f"Current state: {task.state}, expected: {TaskStatusList.BACKLOG}."
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)

    await plan.db_save()
    return task


@pytest.fixture(scope="function")
async def task_awaiting_preparation(plan_step_0: Plan) -> Task:
    """
    Returns a task '300.4' awaiting preparation at plan_step_0.
    Raises an exception with detailed information if conditions are not met.
    """
    plan = plan_step_0  # Added line
    task = await plan.get_task(task_id="300.4")

    if task.state != TaskStatusList.BACKLOG:
        error_message = (
            "Error: Fixture task_awaiting_preparation needs update. "
            f"Current state: {task.state}, expected: {TaskStatusList.BACKLOG}."
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)
    await plan.db_save()
    return task


@pytest.fixture(scope="function")
async def task_with_ongoing_subtasks(plan_step_10: Plan) -> Task:
    plan = plan_step_10  # Added line
    task = await plan.get_task(task_id="300.1")
    predecessors = await task._task_predecessors.get_all_tasks_from_stack()
    done_predecessors = await task._task_predecessors.get_done_tasks_from_stack()
    active_subtasks = await task.subtasks.get_active_tasks_from_stack()

    if len(predecessors) != len(done_predecessors):
        error_message = (
            "Error: Fixture task_with_ongoing_subtasks needs update\n"
            f"Total predecessors: {len(predecessors)}, done predecessors: {len(done_predecessors)}.\n\n\n"
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)

    if len(active_subtasks) == 0:
        error_message = (
            "Error: Fixture task_with_ongoing_subtasks needs update\n"
            f"Active subtasks: {len(active_subtasks)}.\n\n\n"
            f"plan._all_task_ids: {plan._all_task_ids}.\n\n\n"
            f"plan._ready_task_ids: {plan._ready_task_ids}.\n\n\n"
            f"plan._done_task_ids: {plan._done_task_ids}.\n\n\n"
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)

    if task.state != TaskStatusList.IN_PROGRESS_WITH_SUBTASKS:
        error_message = (
            "Error: Fixture task_with_ongoing_subtasks needs update. "
            f"Current state: {task.state}, expected: {TaskStatusList.IN_PROGRESS_WITH_SUBTASKS}."
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)
    await plan.db_save()
    return task


@pytest.fixture(scope="function")
async def task_ready_all_subtasks_done(plan_step_11: Plan) -> Task:
    """
    Returns a task 'task_300_1_gather_ingredients' with all subtasks done.
    Raises an exception if conditions are not met.
    """
    plan = plan_step_11  # Added line
    task = await plan.get_task(task_id="300.1")
    all_subtasks = await task.subtasks.get_all_tasks_from_stack()
    done_subtasks = await task.subtasks.get_done_tasks_from_stack()
    predecessors = await task._task_predecessors.get_all_tasks_from_stack()
    done_predecessors = await task._task_predecessors.get_done_tasks_from_stack()

    if len(all_subtasks) != len(done_subtasks):
        raise Exception(
            f"Error: Fixture task_ready_all_subtasks_done needs update\n\n\n {await make_tree(plan)}"
        )

    if len(predecessors) != len(done_predecessors):
        raise Exception(
            f"Error : This situation should never occur \n\n\n {await make_tree(plan)}"
        )

    await plan.db_save()
    return task


@pytest.fixture(scope="function")
async def task_with_mixed_predecessors(plan_step_19: Plan) -> Task:
    """
    Returns a task with mixed predecessor statuses.
    Raises an exception if conditions are not met.
    """
    plan = plan_step_19  # Added line
    task = await plan.get_task(task_id="300.6")
    active_tasks = await task._task_predecessors.get_active_tasks_from_stack()
    ready_tasks = await task._task_predecessors.get_ready_tasks_from_stack()
    all_tasks = await task._task_predecessors.get_all_tasks_from_stack()
    done_tasks = await task._task_predecessors.get_done_tasks_from_stack()

    if not (active_tasks and len(active_tasks) < len(all_tasks)):
        error_message = (
            f"Error: Fixture task_with_mixed_predecessors needs update\n"
            f"There are {len(active_tasks)} active tasks, \n"
            f"{len(ready_tasks)} ready tasks, \n"
            f"{len(done_tasks)} done tasks, \n"
            f"plan.get_active_tasks_ids()): {plan.get_active_tasks_ids()}.\n"
            f"plan._all_task_ids: {plan._all_task_ids}.\n"
            f"task._task_predecessors._task_ids: {task._task_predecessors._task_ids}.\n"
            f"for a total of {len(all_tasks)} tasks\n\n\n"
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)

    # NOTE : Occune sous tache ne dois démarée tant que il y a dé précécesseur sur la tache actuelle
    done_subtasks = await task.subtasks.get_done_tasks_from_stack()
    if len(done_subtasks) > 0:
        error_message = (
            f"Error: This situation should not occur \n"
            f"There are {len(done_subtasks)} done_subtasks tasks, "
            f"{await make_tree(plan)}"
        )
        raise Exception(error_message)

    await plan.db_save()
    return task
