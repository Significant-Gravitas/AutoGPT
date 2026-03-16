import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, cleanup } from "@testing-library/react";
import { render } from "@/tests/integrations/test-utils";
import React from "react";
import { BlockUIType } from "../components/types";
import type { CustomNodeData } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import type { NodeProps } from "@xyflow/react";

// ---- Mock sub-components ----

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeContainer",
  () => ({
    NodeContainer: ({ children, hasErrors }: any) => (
      <div data-testid="node-container" data-has-errors={String(!!hasErrors)}>
        {children}
      </div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeHeader",
  () => ({
    NodeHeader: ({ data }: any) => (
      <div data-testid="node-header">{data.title}</div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/StickyNoteBlock",
  () => ({
    StickyNoteBlock: ({ data }: any) => (
      <div data-testid="sticky-note-block">{data.title}</div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeAdvancedToggle",
  () => ({
    NodeAdvancedToggle: () => <div data-testid="node-advanced-toggle" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeOutput/NodeOutput",
  () => ({
    NodeDataRenderer: () => <div data-testid="node-data-renderer" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeExecutionBadge",
  () => ({
    NodeExecutionBadge: () => <div data-testid="node-execution-badge" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeRightClickMenu",
  () => ({
    NodeRightClickMenu: ({ children }: any) => (
      <div data-testid="node-right-click-menu">{children}</div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/WebhookDisclaimer",
  () => ({
    WebhookDisclaimer: () => <div data-testid="webhook-disclaimer" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/SubAgentUpdate/SubAgentUpdateFeature",
  () => ({
    SubAgentUpdateFeature: () => <div data-testid="sub-agent-update" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/AyrshareConnectButton",
  () => ({
    AyrshareConnectButton: () => <div data-testid="ayrshare-connect-button" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/FormCreator",
  () => ({
    FormCreator: () => <div data-testid="form-creator" />,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/OutputHandler",
  () => ({
    OutputHandler: () => <div data-testid="output-handler" />,
  }),
);

vi.mock(
  "@/components/renderers/InputRenderer/utils/input-schema-pre-processor",
  () => ({
    preprocessInputSchema: (schema: any) => schema,
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/useCustomNode",
  () => ({
    useCustomNode: ({ data }: { data: CustomNodeData }) => ({
      inputSchema: data.inputSchema,
      outputSchema: data.outputSchema,
      isMCPWithTool: false,
    }),
  }),
);

vi.mock("@xyflow/react", async () => {
  const actual = await vi.importActual("@xyflow/react");
  return {
    ...actual,
    useReactFlow: () => ({
      getNodes: () => [],
      getEdges: () => [],
      setNodes: vi.fn(),
      setEdges: vi.fn(),
      getNode: vi.fn(),
    }),
    useNodeId: () => "test-node-id",
    useUpdateNodeInternals: () => vi.fn(),
    Handle: ({ children }: any) => <div>{children}</div>,
    Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
  };
});

import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";

// ---- Helpers ----

function buildNodeData(
  overrides: Partial<CustomNodeData> = {},
): CustomNodeData {
  return {
    hardcodedValues: {},
    title: "Test Block",
    description: "A test block",
    inputSchema: { type: "object", properties: {} },
    outputSchema: { type: "object", properties: {} },
    uiType: BlockUIType.STANDARD,
    block_id: "block-123",
    costs: [],
    categories: [],
    ...overrides,
  };
}

function buildNodeProps(
  dataOverrides: Partial<CustomNodeData> = {},
  propsOverrides: Partial<NodeProps<any>> = {},
): any {
  return {
    id: "node-1",
    data: buildNodeData(dataOverrides),
    selected: false,
    type: "custom",
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
    zIndex: 0,
    dragging: false,
    dragHandle: undefined,
    selectable: true,
    deletable: true,
    parentId: undefined,
    ...propsOverrides,
  };
}

function renderCustomNode(
  dataOverrides: Partial<CustomNodeData> = {},
  propsOverrides: Partial<NodeProps<any>> = {},
) {
  const props = buildNodeProps(dataOverrides, propsOverrides);
  return render(<CustomNode {...props} />);
}

// ---- Tests ----

beforeEach(() => {
  cleanup();
});

describe("CustomNode", () => {
  describe("STANDARD type rendering", () => {
    it("renders NodeHeader with the block title", () => {
      renderCustomNode({ title: "My Standard Block" });

      const header = screen.getByTestId("node-header");
      expect(header).toBeDefined();
      expect(header.textContent).toContain("My Standard Block");
    });

    it("renders NodeContainer, FormCreator, OutputHandler, and NodeExecutionBadge", () => {
      renderCustomNode();

      expect(screen.getByTestId("node-container")).toBeDefined();
      expect(screen.getByTestId("form-creator")).toBeDefined();
      expect(screen.getByTestId("output-handler")).toBeDefined();
      expect(screen.getByTestId("node-execution-badge")).toBeDefined();
      expect(screen.getByTestId("node-data-renderer")).toBeDefined();
      expect(screen.getByTestId("node-advanced-toggle")).toBeDefined();
    });

    it("wraps content in NodeRightClickMenu", () => {
      renderCustomNode();

      expect(screen.getByTestId("node-right-click-menu")).toBeDefined();
    });

    it("does not render StickyNoteBlock for STANDARD type", () => {
      renderCustomNode();

      expect(screen.queryByTestId("sticky-note-block")).toBeNull();
    });
  });

  describe("NOTE type rendering", () => {
    it("renders StickyNoteBlock instead of main UI", () => {
      renderCustomNode({ uiType: BlockUIType.NOTE, title: "My Note" });

      const note = screen.getByTestId("sticky-note-block");
      expect(note).toBeDefined();
      expect(note.textContent).toContain("My Note");
    });

    it("does not render NodeContainer or other standard components", () => {
      renderCustomNode({ uiType: BlockUIType.NOTE });

      expect(screen.queryByTestId("node-container")).toBeNull();
      expect(screen.queryByTestId("node-header")).toBeNull();
      expect(screen.queryByTestId("form-creator")).toBeNull();
      expect(screen.queryByTestId("output-handler")).toBeNull();
    });
  });

  describe("WEBHOOK type rendering", () => {
    it("renders WebhookDisclaimer for WEBHOOK type", () => {
      renderCustomNode({ uiType: BlockUIType.WEBHOOK });

      expect(screen.getByTestId("webhook-disclaimer")).toBeDefined();
    });

    it("renders WebhookDisclaimer for WEBHOOK_MANUAL type", () => {
      renderCustomNode({ uiType: BlockUIType.WEBHOOK_MANUAL });

      expect(screen.getByTestId("webhook-disclaimer")).toBeDefined();
    });
  });

  describe("AGENT type rendering", () => {
    it("renders SubAgentUpdateFeature for AGENT type", () => {
      renderCustomNode({ uiType: BlockUIType.AGENT });

      expect(screen.getByTestId("sub-agent-update")).toBeDefined();
    });

    it("does not render SubAgentUpdateFeature for non-AGENT types", () => {
      renderCustomNode({ uiType: BlockUIType.STANDARD });

      expect(screen.queryByTestId("sub-agent-update")).toBeNull();
    });
  });

  describe("OUTPUT type rendering", () => {
    it("does not render OutputHandler for OUTPUT type", () => {
      renderCustomNode({ uiType: BlockUIType.OUTPUT });

      expect(screen.queryByTestId("output-handler")).toBeNull();
    });

    it("still renders FormCreator and other components for OUTPUT type", () => {
      renderCustomNode({ uiType: BlockUIType.OUTPUT });

      expect(screen.getByTestId("form-creator")).toBeDefined();
      expect(screen.getByTestId("node-header")).toBeDefined();
      expect(screen.getByTestId("node-execution-badge")).toBeDefined();
    });
  });

  describe("AYRSHARE type rendering", () => {
    it("renders AyrshareConnectButton for AYRSHARE type", () => {
      renderCustomNode({ uiType: BlockUIType.AYRSHARE });

      expect(screen.getByTestId("ayrshare-connect-button")).toBeDefined();
    });

    it("does not render AyrshareConnectButton for non-AYRSHARE types", () => {
      renderCustomNode({ uiType: BlockUIType.STANDARD });

      expect(screen.queryByTestId("ayrshare-connect-button")).toBeNull();
    });
  });

  describe("error states", () => {
    it("sets hasErrors on NodeContainer when data.errors has non-empty values", () => {
      renderCustomNode({
        errors: { field1: "This field is required" },
      });

      const container = screen.getByTestId("node-container");
      expect(container.getAttribute("data-has-errors")).toBe("true");
    });

    it("does not set hasErrors when data.errors is empty", () => {
      renderCustomNode({ errors: {} });

      const container = screen.getByTestId("node-container");
      expect(container.getAttribute("data-has-errors")).toBe("false");
    });

    it("does not set hasErrors when data.errors values are all empty strings", () => {
      renderCustomNode({ errors: { field1: "" } });

      const container = screen.getByTestId("node-container");
      expect(container.getAttribute("data-has-errors")).toBe("false");
    });

    it("sets hasErrors when last execution result has error in output_data", () => {
      renderCustomNode({
        nodeExecutionResults: [
          {
            output_data: { error: "Something went wrong" },
          } as any,
        ],
      });

      const container = screen.getByTestId("node-container");
      expect(container.getAttribute("data-has-errors")).toBe("true");
    });

    it("does not set hasErrors when execution results have no error", () => {
      renderCustomNode({
        nodeExecutionResults: [
          {
            output_data: { result: "success" },
          } as any,
        ],
      });

      const container = screen.getByTestId("node-container");
      expect(container.getAttribute("data-has-errors")).toBe("false");
    });
  });

  describe("NodeExecutionBadge", () => {
    it("always renders NodeExecutionBadge for non-NOTE types", () => {
      renderCustomNode({ uiType: BlockUIType.STANDARD });
      expect(screen.getByTestId("node-execution-badge")).toBeDefined();
    });

    it("renders NodeExecutionBadge for AGENT type", () => {
      renderCustomNode({ uiType: BlockUIType.AGENT });
      expect(screen.getByTestId("node-execution-badge")).toBeDefined();
    });

    it("renders NodeExecutionBadge for OUTPUT type", () => {
      renderCustomNode({ uiType: BlockUIType.OUTPUT });
      expect(screen.getByTestId("node-execution-badge")).toBeDefined();
    });
  });

  describe("edge cases", () => {
    it("renders without nodeExecutionResults", () => {
      renderCustomNode({ nodeExecutionResults: undefined });

      const container = screen.getByTestId("node-container");
      expect(container).toBeDefined();
      expect(container.getAttribute("data-has-errors")).toBe("false");
    });

    it("renders without errors property", () => {
      renderCustomNode({ errors: undefined });

      const container = screen.getByTestId("node-container");
      expect(container).toBeDefined();
      expect(container.getAttribute("data-has-errors")).toBe("false");
    });

    it("renders with empty execution results array", () => {
      renderCustomNode({ nodeExecutionResults: [] });

      const container = screen.getByTestId("node-container");
      expect(container).toBeDefined();
      expect(container.getAttribute("data-has-errors")).toBe("false");
    });
  });
});
