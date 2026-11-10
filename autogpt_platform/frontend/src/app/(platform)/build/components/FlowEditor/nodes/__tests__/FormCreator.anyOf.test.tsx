import type { RJSFSchema } from "@rjsf/utils";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useHistoryStore } from "@/app/(platform)/build/stores/historyStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { BlockUIType } from "../../../types";
import type { CustomNode } from "../CustomNode/CustomNode";
import { FormCreator } from "../FormCreator";

const schema: RJSFSchema = {
  type: "object",
  properties: {
    payload: { title: "Payload" },
    other: { type: "string", title: "Other" },
  },
};

function makeNode(
  hardcodedValues: Record<string, unknown>,
  uiType: BlockUIType,
): CustomNode {
  return {
    id: "anyof-node",
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      title: "AnyOf node",
      description: "",
      block_id: "test-block",
      costs: [],
      categories: [],
      hardcodedValues,
      uiType,
      inputSchema: schema,
      outputSchema: {
        type: "object",
        properties: {
          response: { type: "object", additionalProperties: true },
        },
      },
    },
  };
}

function form(uiType: BlockUIType) {
  return (
    <FormCreator
      jsonSchema={schema}
      nodeId="anyof-node"
      uiType={uiType}
      isMCPWithTool={uiType === BlockUIType.MCP_TOOL}
      showHandles={false}
    />
  );
}

function values() {
  return useNodeStore.getState().getHardCodedValues("anyof-node");
}

function setup(
  hardcodedValues: Record<string, unknown>,
  uiType = BlockUIType.STANDARD,
) {
  useNodeStore.setState({
    nodes: [makeNode(hardcodedValues, uiType)],
    nodeAdvancedStates: {},
  });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });
  return render(form(uiType));
}

afterEach(() => {
  useNodeStore.setState({ nodes: [], nodeAdvancedStates: {} });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });
});

describe("FormCreator anyOf input storage", () => {
  it.each([BlockUIType.STANDARD, BlockUIType.AGENT, BlockUIType.MCP_TOOL])(
    "edits and remounts an empty dictionary for %s without moving its stored values",
    async (uiType) => {
      const initial =
        uiType === BlockUIType.AGENT
          ? { inputs: { payload: {} }, marker: "keep" }
          : uiType === BlockUIType.MCP_TOOL
            ? { tool_arguments: { payload: {} }, marker: "keep" }
            : { payload: {} };
      const { unmount } = setup(initial, uiType);
      expect(screen.getByRole("combobox").textContent).toContain("object");
      expect(values()).toEqual(initial);
      fireEvent.click(screen.getByRole("button", { name: /^add item$/i }));
      fireEvent.change(await screen.findByDisplayValue("New Value"), {
        target: { value: "Saved value" },
      });

      const expected =
        uiType === BlockUIType.AGENT
          ? { inputs: { payload: { newKey: "Saved value" } }, marker: "keep" }
          : uiType === BlockUIType.MCP_TOOL
            ? {
                tool_arguments: { payload: { newKey: "Saved value" } },
                marker: "keep",
              }
            : { payload: { newKey: "Saved value" } };
      await waitFor(() => expect(values()).toEqual(expected));
      unmount();
      render(form(uiType));
      expect(await screen.findByDisplayValue("Saved value")).toBeDefined();
      expect(screen.getByRole("combobox").textContent).toContain("object");
    },
  );

  it("switches type while retaining unrelated inputs and dynamic output edges", async () => {
    const user = userEvent.setup();
    const initial = { payload: {}, other: "keep" };
    setup(initial);
    const outputSchema = useNodeStore.getState().nodes[0].data.outputSchema;
    act(() => {
      const edgeStore = useEdgeStore.getState();
      edgeStore.addEdge({
        source: "anyof-node",
        sourceHandle: "response_#_reviews_#_author",
        target: "destination",
        targetHandle: "input",
      });
    });
    const outputEdge = useEdgeStore.getState().edges[0];

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "string" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Payload" }), {
      target: { value: "Text value" },
    });

    await waitFor(() =>
      expect(values()).toEqual({ payload: "Text value", other: "keep" }),
    );
    expect(useEdgeStore.getState().edges).toEqual([outputEdge]);
    expect(useNodeStore.getState().nodes[0].data.outputSchema).toEqual(
      outputSchema,
    );
    expect(screen.getByRole("combobox").textContent).toContain("string");
  });
});
