import { describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup } from "@testing-library/react";
import { render } from "@/tests/integrations/test-utils";
import React from "react";
import type { RJSFSchema } from "@rjsf/utils";
import type { NodeProps } from "@xyflow/react";
import { BlockUIType } from "../components/types";
import type {
  CustomNodeData,
  CustomNode as CustomNodeType,
} from "../components/FlowEditor/nodes/CustomNode/CustomNode";

const formCreatorSchemas: RJSFSchema[] = [];

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeContainer",
  () => ({
    NodeContainer: ({ children }: { children: React.ReactNode }) => (
      <div>{children}</div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeHeader",
  () => ({ NodeHeader: () => null }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeAdvancedToggle",
  () => ({ NodeAdvancedToggle: () => null }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeOutput/NodeOutput",
  () => ({ NodeDataRenderer: () => null }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeExecutionBadge",
  () => ({ NodeExecutionBadge: () => null }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/NodeRightClickMenu",
  () => ({
    NodeRightClickMenu: ({ children }: { children: React.ReactNode }) => (
      <div>{children}</div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/OutputHandler",
  () => ({ OutputHandler: () => null }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/FormCreator",
  () => ({
    FormCreator: ({ jsonSchema }: { jsonSchema: RJSFSchema }) => {
      formCreatorSchemas.push(jsonSchema);
      return <div data-testid="form-creator" />;
    },
  }),
);

import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";

const inputSchema: RJSFSchema = {
  type: "object",
  properties: {
    prompt: { type: "string", title: "Prompt" },
    anything: { title: "Anything" },
  },
};

function buildProps(
  data: CustomNodeData,
  overrides: Partial<NodeProps<CustomNodeType>> = {},
): NodeProps<CustomNodeType> {
  return {
    id: "node-1",
    data,
    selected: false,
    type: "custom",
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
    zIndex: 0,
    dragging: false,
    dragHandle: undefined,
    draggable: true,
    selectable: true,
    deletable: true,
    parentId: undefined,
    width: undefined,
    height: undefined,
    sourcePosition: undefined,
    targetPosition: undefined,
    ...overrides,
  };
}

beforeEach(() => {
  cleanup();
  formCreatorSchemas.length = 0;
});

describe("CustomNode input schema identity", () => {
  const data: CustomNodeData = {
    hardcodedValues: {},
    title: "Test Block",
    description: "A test block",
    inputSchema,
    outputSchema: { type: "object", properties: {} },
    uiType: BlockUIType.STANDARD,
    block_id: "block-123",
    costs: [],
    categories: [],
  };

  it("passes the same jsonSchema object to FormCreator while the node is dragged", () => {
    const { rerender } = render(<CustomNode {...buildProps(data)} />);

    rerender(
      <CustomNode
        {...buildProps(data, {
          positionAbsoluteX: 40,
          positionAbsoluteY: 20,
          dragging: true,
        })}
      />,
    );

    expect(formCreatorSchemas).toHaveLength(2);
    expect(formCreatorSchemas[1]).toBe(formCreatorSchemas[0]);
  });

  it("still preprocesses the schema it passes on", () => {
    render(<CustomNode {...buildProps(data)} />);

    const anything = formCreatorSchemas[0].properties?.anything as RJSFSchema;
    expect(anything.anyOf).toBeDefined();
  });

  it("passes a new jsonSchema when the input schema changes", () => {
    const { rerender } = render(<CustomNode {...buildProps(data)} />);

    rerender(
      <CustomNode
        {...buildProps({
          ...data,
          inputSchema: {
            ...inputSchema,
            properties: { other: { type: "number", title: "Other" } },
          },
        })}
      />,
    );

    expect(formCreatorSchemas).toHaveLength(2);
    expect(formCreatorSchemas[1]).not.toBe(formCreatorSchemas[0]);
    expect(formCreatorSchemas[1].properties).toHaveProperty("other");
  });
});
