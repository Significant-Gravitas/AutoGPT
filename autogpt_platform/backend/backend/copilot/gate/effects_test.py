from types import SimpleNamespace

import pytest

from backend.blocks import get_blocks
from backend.blocks._base import Block, BlockEffect

from .effects import UNRESOLVABLE_BLOCKS, block_effect, graph_effect


def _block(name: str, effect: BlockEffect | None) -> Block:
    """A stand-in with the two attributes ``block_effect`` reads."""
    return type(name, (SimpleNamespace,), {})(effect=effect)


def _graph(*nodes, links=()):
    return SimpleNamespace(nodes=list(nodes), links=list(links))


def _node(node_id: str, block, **input_default):
    return SimpleNamespace(id=node_id, block=block, input_default=input_default)


@pytest.mark.parametrize(
    "effect", [BlockEffect.NONE, BlockEffect.READ, BlockEffect.WRITE, None]
)
def test_declaration_wins_for_an_ordinary_block(effect):
    assert block_effect(_block("OrdinaryBlock", effect), {}) is effect


def test_web_request_is_read_only_for_a_constant_safe_method():
    web = _block("SendWebRequestBlock", None)
    assert block_effect(web, {"method": "GET"}) is BlockEffect.READ
    assert block_effect(web, {"method": "head"}) is BlockEffect.READ
    assert block_effect(web, {"method": "POST"}) is BlockEffect.WRITE
    # The field's own default is POST, so an absent method is a write.
    assert block_effect(web, {}) is BlockEffect.WRITE


def test_web_request_with_a_linked_method_is_unresolvable():
    web = _block("SendAuthenticatedWebRequestBlock", None)
    assert block_effect(web, {"method": "GET"}, linked_inputs={"method"}) is None


def test_sql_query_follows_read_only_and_its_default():
    sql = _block("SQLQueryBlock", None)
    assert block_effect(sql, {}) is BlockEffect.READ
    assert block_effect(sql, {"read_only": True}) is BlockEffect.READ
    assert block_effect(sql, {"read_only": False}) is BlockEffect.WRITE
    assert block_effect(sql, {"read_only": True}, linked_inputs=["read_only"]) is None


def test_graph_takes_the_worst_node():
    pure = _block("PureBlock", BlockEffect.NONE)
    read = _block("ReadBlock", BlockEffect.READ)
    write = _block("WriteBlock", BlockEffect.WRITE)

    assert graph_effect([_graph(_node("a", pure))]) is BlockEffect.NONE
    assert (
        graph_effect([_graph(_node("a", pure), _node("b", read))]) is BlockEffect.READ
    )
    assert (
        graph_effect([_graph(_node("a", read), _node("b", write))]) is BlockEffect.WRITE
    )


def test_a_write_in_a_sub_graph_makes_the_whole_run_a_write():
    read = _block("ReadBlock", BlockEffect.READ)
    write = _block("WriteBlock", BlockEffect.WRITE)
    parent = _graph(_node("a", read))
    sub = _graph(_node("b", write))

    assert graph_effect([parent]) is BlockEffect.READ
    assert graph_effect([parent, sub]) is BlockEffect.WRITE


def test_one_unresolvable_node_makes_the_whole_run_unresolvable():
    read = _block("ReadBlock", BlockEffect.READ)
    unknown = _block("UnclassifiedBlock", None)
    assert graph_effect([_graph(_node("a", read), _node("b", unknown))]) is None


def test_a_linked_method_inside_a_graph_makes_it_unresolvable():
    web = _block("SendWebRequestBlock", None)
    node = _node("a", web, method="GET")
    link = SimpleNamespace(sink_id="a", sink_name="method")

    assert graph_effect([_graph(node)]) is BlockEffect.READ
    assert graph_effect([_graph(node, links=[link])]) is None


def test_input_aware_and_unresolvable_names_are_real_registered_blocks():
    """A rename must not silently turn one of these into an ordinary block."""
    from .effects import _INPUT_AWARE

    registered = {cls.__name__ for cls in get_blocks().values()}
    assert not (set(_INPUT_AWARE) - registered)
    assert not (UNRESOLVABLE_BLOCKS - registered)


def test_unresolvable_blocks_declare_no_effect():
    from .effects import _INPUT_AWARE

    by_name = {cls.__name__: cls for cls in get_blocks().values()}
    for name in UNRESOLVABLE_BLOCKS | set(_INPUT_AWARE):
        assert (
            by_name[name]().effect is None
        ), f"{name} resolves its effect at call time, so it must not declare one."
