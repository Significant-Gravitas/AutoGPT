// Graphs that do real work in the executor with no model and no
// credentials: deterministic blocks, zero credits per block, a known number
// of node executions per run. The seed creates them under each account; the
// graph journey runs them.
//
// Block ids and pin names are the backend's (backend/blocks/io.py, text.py,
// iteration.py, maths.py, data_manipulation.py, branching.py). The server
// assigns new ids to the graph and its nodes; node ids here only wire links.

const BLOCKS = {
  agentInput: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  agentOutput: "363ae599-353e-4804-937e-b2ee3cef3da4",
  fillTemplate: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
  combineTexts: "e30a4d42-7b7d-4e6a-b36e-1f9b8e3b7d85",
  textSplit: "d5ea33c8-a575-477a-b42f-2fe3be5055ec",
  textReplace: "7e7c87ab-3469-4bcc-9abe-67705091b713",
  calculator: "b1ab9b19-67a6-406d-abf5-2dba76d00c79",
  countItems: "3c9c2f42-b0c3-435f-ba35-05f7a25c772a",
  stepThroughItems: "f66a3543-28d3-4ab5-8945-9b336371e2ce",
  storeValue: "1ff065e9-88e8-4358-9d82-8dc91f622ba9",
  createDictionary: "b924ddf4-de4f-4b56-9a85-358930dcbc91",
  condition: "715696a0-e1da-45c8-b209-c2fa9c3b0be6",
};

function node(id, blockId, inputDefault, x) {
  return {
    id,
    block_id: blockId,
    input_default: inputDefault || {},
    metadata: { position: { x: x || 0, y: 0 } },
  };
}

function link(sourceId, sourceName, sinkId, sinkName) {
  return {
    source_id: sourceId,
    source_name: sourceName,
    sink_id: sinkId,
    sink_name: sinkName,
  };
}

// Splits a comma-separated input, formats every item, counts them and
// reports both: a fan-out through the loop block, about a dozen node
// executions for three items.
function splitAndFormat(name) {
  return {
    name,
    description: "Load test: split, loop, format, count",
    nodes: [
      node("in", BLOCKS.agentInput, { name: "items", title: "Items" }, 0),
      node("split", BLOCKS.textSplit, { delimiter: ",", strip: true }, 200),
      node("loop", BLOCKS.stepThroughItems, {}, 400),
      node("fmt", BLOCKS.fillTemplate, { format: "item={{ item }}" }, 600),
      node("count", BLOCKS.countItems, {}, 400),
      node(
        "out",
        BLOCKS.agentOutput,
        { name: "formatted", title: "Formatted" },
        800,
      ),
      node(
        "outCount",
        BLOCKS.agentOutput,
        { name: "count", title: "Count" },
        800,
      ),
    ],
    links: [
      link("in", "result", "split", "text"),
      link("split", "texts", "loop", "items"),
      link("loop", "item", "fmt", "values_#_item"),
      link("fmt", "output", "out", "value"),
      link("split", "texts", "count", "collection"),
      link("count", "count", "outCount", "value"),
    ],
    inputs: { items: "alpha,beta,gamma" },
  };
}

// Two numeric inputs through a chain of calculations and a comparison: a
// linear pipeline of about eight node executions.
function arithmetic(name) {
  return {
    name,
    description: "Load test: calculate, compare, format",
    nodes: [
      node("a", BLOCKS.agentInput, { name: "a", title: "A" }, 0),
      node("b", BLOCKS.agentInput, { name: "b", title: "B" }, 0),
      node("sum", BLOCKS.calculator, { operation: "Add" }, 200),
      node("product", BLOCKS.calculator, { operation: "Multiply" }, 200),
      node(
        "cmp",
        BLOCKS.condition,
        { operator: ">", yes_value: "product wins", no_value: "sum wins" },
        400,
      ),
      node("fmt", BLOCKS.fillTemplate, { format: "{{ verdict }}" }, 600),
      node(
        "out",
        BLOCKS.agentOutput,
        { name: "verdict", title: "Verdict" },
        800,
      ),
      node("outSum", BLOCKS.agentOutput, { name: "sum", title: "Sum" }, 800),
    ],
    links: [
      link("a", "result", "sum", "a"),
      link("b", "result", "sum", "b"),
      link("a", "result", "product", "a"),
      link("b", "result", "product", "b"),
      link("product", "result", "cmp", "value1"),
      link("sum", "result", "cmp", "value2"),
      link("cmp", "result", "fmt", "values_#_verdict"),
      link("fmt", "output", "out", "value"),
      link("sum", "result", "outSum", "value"),
    ],
    inputs: { a: 7, b: 6 },
  };
}

// Text transformations and a dictionary: replace, combine, build a record,
// format it; about seven node executions.
function textPipeline(name) {
  return {
    name,
    description: "Load test: replace, combine, record, format",
    nodes: [
      node("in", BLOCKS.agentInput, { name: "subject", title: "Subject" }, 0),
      node(
        "replace",
        BLOCKS.textReplace,
        { old: "world", new: "platform" },
        200,
      ),
      node("combine", BLOCKS.combineTexts, { delimiter: " " }, 400),
      node(
        "record",
        BLOCKS.createDictionary,
        { values: { source: "load-test" } },
        400,
      ),
      node("keep", BLOCKS.storeValue, {}, 600),
      node(
        "fmt",
        BLOCKS.fillTemplate,
        { format: "{{ text }} ({{ record.source }})" },
        600,
      ),
      node(
        "out",
        BLOCKS.agentOutput,
        { name: "summary", title: "Summary" },
        800,
      ),
    ],
    links: [
      link("in", "result", "replace", "text"),
      link("replace", "output", "combine", "input_$_0"),
      link("in", "result", "combine", "input_$_1"),
      link("combine", "output", "keep", "input"),
      link("keep", "output", "fmt", "values_#_text"),
      link("record", "dictionary", "fmt", "values_#_record"),
      link("fmt", "output", "out", "value"),
    ],
    inputs: { subject: "hello world" },
  };
}

const TEMPLATES = [splitAndFormat, arithmetic, textPipeline];

// The n-th graph of a run's account: the templates in turn, named so every
// graph of a run can be found.
export function seedGraph(runId, accountIndex, index) {
  const template = TEMPLATES[index % TEMPLATES.length];
  const graph = template(`Load test ${runId} a${accountIndex} g${index + 1}`);
  return {
    body: {
      graph: {
        name: graph.name,
        description: graph.description,
        nodes: graph.nodes,
        links: graph.links,
      },
      source: "builder",
    },
    inputs: graph.inputs,
  };
}

export const TEMPLATE_COUNT = TEMPLATES.length;
