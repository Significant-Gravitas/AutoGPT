import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { BlockUIType } from "@/lib/autogpt-server-api";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

import { Block } from "../Block";

const mocks = vi.hoisted(() => ({
  addBlock: vi.fn(() => ({
    id: "node-1",
    position: { x: 0, y: 0 },
    data: { metadata: undefined },
  })),
  updateNodeData: vi.fn(),
  setViewport: vi.fn(),
}));

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({ setViewport: mocks.setViewport }),
}));

vi.mock("@/app/(platform)/build/stores/nodeStore", () => ({
  useNodeStore: (selector?: (state: unknown) => unknown) => {
    const state = {
      addBlock: mocks.addBlock,
      updateNodeData: mocks.updateNodeData,
    };
    return selector ? selector(state) : state;
  },
}));

vi.mock("@/app/(platform)/build/stores/controlPanelStore", () => ({
  useControlPanelStore: (selector: (state: unknown) => unknown) =>
    selector({ setBlockMenuOpen: vi.fn() }),
}));

vi.mock("@/app/(platform)/build/components/MCPToolDialog", () => ({
  MCPToolDialog: ({
    open,
    onConfirm,
  }: {
    open: boolean;
    onConfirm: (result: unknown) => void;
  }) =>
    open ? (
      <button
        onClick={() =>
          onConfirm({
            serverUrl: "https://mcp.example.com/mcp",
            serverName: "Example MCP",
            selectedTool: "lookup_item",
            toolInputSchema: { type: "object" },
            availableTools: {},
            credentials: {
              id: "credential-1",
              provider: "mcp",
              type: "oauth2",
              title: "MCP: mcp.example.com",
            },
          })
        }
      >
        Confirm MCP tool
      </button>
    ) : null,
}));

describe("MCP block creation", () => {
  it("keeps the node executable when binding a discovered credential", () => {
    const blockData = {
      id: "mcp-tool-block",
      name: "MCP Tool Block",
      description: "Run an MCP tool",
      uiType: BlockUIType.MCP_TOOL,
      inputSchema: {},
      outputSchema: {},
      costs: [],
      categories: [],
      contributors: [],
      staticOutput: false,
    } as BlockInfo;

    render(<Block blockData={blockData} title={blockData.name} />);
    fireEvent.click(screen.getByRole("button", { name: /mcp tool/i }));
    fireEvent.click(screen.getByRole("button", { name: /confirm mcp tool/i }));

    expect(mocks.addBlock).toHaveBeenCalledWith(
      blockData,
      expect.objectContaining({
        credentials: expect.objectContaining({ id: "credential-1" }),
      }),
    );
    const metadata = mocks.updateNodeData.mock.calls[0]?.[1]?.metadata;
    expect(metadata).toEqual({ customized_name: "Example MCP: Lookup Item" });
    expect(metadata?.credentials_optional).toBeUndefined();
  });
});
