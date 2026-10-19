import { BlockUIType } from "@/app/(platform)/build/components/types";
import { FormCreator } from "@/app/(platform)/build/components/FlowEditor/nodes/FormCreator";
import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useHistoryStore } from "@/app/(platform)/build/stores/historyStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { render } from "@/tests/integrations/test-utils";
import { act, screen, within } from "@testing-library/react";
import type { RJSFSchema } from "@rjsf/utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const fieldRenders = vi.hoisted(() => [] as string[]);

// FieldError renders once per FieldTemplate render and is not memoised, so it
// counts how often each node's field templates re-render.
vi.mock(
  "@/components/renderers/InputRenderer/base/standard/FieldError",
  () => ({
    FieldError: ({ nodeId, fieldId }: { nodeId: string; fieldId: string }) => {
      fieldRenders.push(`${nodeId}:${fieldId}`);
      return null;
    },
  }),
);

const schema = {
  type: "object",
  properties: {
    prompt: { title: "Prompt", type: "string" },
    topic: { title: "Topic", type: "string" },
  },
} as RJSFSchema;

function makeNode(id: string, hardcodedValues = {}): CustomNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues,
      title: id,
      description: "",
      inputSchema: schema,
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "block-1",
      costs: [],
      categories: [],
    },
  } as unknown as CustomNode;
}

function renderNodes(...ids: string[]) {
  useNodeStore.setState({
    nodes: ids.map((id) => makeNode(id)),
    nodeAdvancedStates: {},
  });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });

  return render(
    <>
      {ids.map((id) => (
        <div key={id} data-testid={id}>
          <FormCreator
            jsonSchema={schema}
            nodeId={id}
            uiType={BlockUIType.STANDARD}
            showHandles={false}
          />
        </div>
      ))}
    </>,
  );
}

function rendersOf(nodeId: string) {
  return fieldRenders.filter((entry) => entry.startsWith(`${nodeId}:`));
}

beforeEach(() => {
  fieldRenders.length = 0;
});

describe("FormCreator re-renders", () => {
  it("does not re-render node A's fields when an edge of node B changes", async () => {
    renderNodes("node-a", "node-b");
    await screen.findAllByText("Topic");
    fieldRenders.length = 0;

    act(() => {
      useEdgeStore.getState().addEdge({
        source: "node-c",
        sourceHandle: "result",
        target: "node-b",
        targetHandle: "prompt",
      });
    });

    expect(rendersOf("node-a")).toEqual([]);
    expect(rendersOf("node-b")).toContain("node-b:prompt");
  });

  it("hides a field's input once an edge connects to it", async () => {
    renderNodes("node-a", "node-b");
    const nodeB = within(screen.getByTestId("node-b"));
    await nodeB.findByText("Topic");
    expect(nodeB.getAllByRole("textbox")).toHaveLength(2);

    act(() => {
      useEdgeStore.getState().addEdge({
        source: "node-c",
        sourceHandle: "result",
        target: "node-b",
        targetHandle: "prompt",
      });
    });

    expect(nodeB.getAllByRole("textbox")).toHaveLength(1);
    expect(
      within(screen.getByTestId("node-a")).getAllByRole("textbox"),
    ).toHaveLength(2);
  });

  it("shows hardcoded values written to the store without a parent re-render", async () => {
    renderNodes("node-a");
    const nodeA = within(screen.getByTestId("node-a"));
    await nodeA.findByText("Topic");

    act(() => {
      useNodeStore
        .getState()
        .updateNodeData("node-a", { hardcodedValues: { topic: "Rivers" } });
    });

    expect(await nodeA.findByDisplayValue("Rivers")).toBeDefined();
  });
});
