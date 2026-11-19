import { fireEvent, screen, within } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { NewControlPanel } from "../components/NewControlPanel/NewControlPanel";
import { FormCreator } from "../components/FlowEditor/nodes/FormCreator";
import type { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "../components/types";
import { useHistoryStore } from "../stores/historyStore";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";

vi.mock(
  "../components/NewControlPanel/NewBlockMenu/BlockMenu/BlockMenu",
  () => ({ BlockMenu: () => null }),
);
vi.mock("../components/NewControlPanel/NewSaveControl/NewSaveControl", () => ({
  NewSaveControl: () => null,
}));
vi.mock(
  "../components/NewControlPanel/NewSearchGraph/GraphMenu/GraphMenu",
  () => ({ GraphSearchMenu: () => null }),
);

const NODE_ID = "node-1";

function textNode(text: string): CustomNode {
  return {
    id: NODE_ID,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      title: "Text block",
      description: "",
      hardcodedValues: { text },
      inputSchema: {},
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "test-block",
      costs: [],
      categories: [],
    },
  };
}

function savedText() {
  return useNodeStore.getState().nodes[0]?.data.hardcodedValues.text;
}

function pressUndo(target: Element) {
  return fireEvent.keyDown(target, {
    key: "z",
    ctrlKey: true,
    metaKey: true,
  });
}

function renderBuilder() {
  return render(
    <ReactFlowProvider>
      <NewControlPanel />
      <FormCreator
        jsonSchema={{
          type: "object",
          properties: { text: { type: "string", title: "Text" } },
        }}
        nodeId={NODE_ID}
        uiType={BlockUIType.STANDARD}
        showHandles={false}
      />
    </ReactFlowProvider>,
  );
}

beforeEach(() => {
  useEdgeStore.setState({ edges: [] });
  useNodeStore.setState({ nodes: [textNode("before")] });
  useHistoryStore.getState().initializeHistory();
  useNodeStore.setState({ nodes: [textNode("after")] });
});

afterEach(() => {
  useHistoryStore.getState().clear();
  useNodeStore.setState({ nodes: [] });
});

describe("builder undo shortcut", () => {
  it("leaves undo to the expanded text box and keeps the workflow as it is", async () => {
    renderBuilder();

    fireEvent.click(screen.getByRole("button", { name: "Expand input" }));
    const expanded = within(await screen.findByRole("dialog")).getByRole(
      "textbox",
    );
    fireEvent.change(expanded, { target: { value: "after, edited" } });

    const notPrevented = pressUndo(expanded);

    expect(notPrevented).toBe(true);
    expect(savedText()).toBe("after");
    expect(useHistoryStore.getState().canRedo()).toBe(false);
  });

  it("leaves undo to the inline text field on the node", () => {
    renderBuilder();

    const notPrevented = pressUndo(screen.getByDisplayValue("after"));

    expect(notPrevented).toBe(true);
    expect(savedText()).toBe("after");
  });

  it("still undoes the workflow when focus is on the canvas", () => {
    renderBuilder();

    const notPrevented = pressUndo(document.body);

    expect(notPrevented).toBe(false);
    expect(savedText()).toBe("before");
  });
});
