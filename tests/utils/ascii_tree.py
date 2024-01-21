import pytest

from AFAAS.interfaces.task.base import AbstractBaseTask

"""""
THE  FUNCTION print_tree(node, prefix="") IS USED TO PRINT A TREE STRUCTURE OF THE TESTS
""" ""


async def make_tree(node: AbstractBaseTask, prefix=""):
    # Print the current node's name
    tree_str = prefix + "|-- " + node.task_goal + " " + node.task_id

    if hasattr(node, "state"):
        tree_str += " " + node.state

    tree_str += "\n"
    # Check if the node has subtasks
    if node.subtasks:
        for i, child_id in enumerate(node.subtasks):
            # If the child is the last child, don't draw the vertical connector
            extension = "    " if i == len(node.subtasks) - 1 else "|   "

            # Get the child task
            child_task = await node.agent.plan.get_task(child_id)

            # Recursively build the tree for each child
            tree_str += await make_tree(child_task, prefix + extension)

    return tree_str


async def print_tree(node: AbstractBaseTask, prefix=""):
    print(await make_tree(node=node, prefix=prefix))


test_trees = {}


async def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add additional section in terminal summary reporting."""
    terminalreporter.write_sep("-", "Tree Structure for Tests")
    for test_name, tree in test_trees.items():
        terminalreporter.write(f"Tree for {test_name}:\n")
        await print_tree(tree, file=terminalreporter)


# Example test using the fixture
def example(plan_step_0):
    # Perform your test logic...
    # Store the tree for reporting
    assert 1 == 1
    test_trees["test_example"] = plan_step_0
