import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { BlockUIType, SpecialBlockID } from "@/lib/autogpt-server-api";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Block } from "./Block";

vi.mock("@/app/(platform)/build/components/MCPToolDialog", () => ({
  MCPToolDialog: ({
    open,
    onConfirm,
  }: {
    open: boolean;
    onConfirm: (result: {
      serverUrl: string;
      serverName: string | null;
      selectedTool: string;
      toolInputSchema: Record<string, unknown>;
      availableTools: Record<string, unknown>;
      credentials: unknown;
    }) => void;
  }) =>
    open ? (
      <button
        onClick={() =>
          onConfirm({
            serverUrl: "http://localhost:8000",
            serverName: "my-mcp",
            selectedTool: "search",
            toolInputSchema: {},
            availableTools: {},
            credentials: null,
          })
        }
      >
        Confirm MCP tool
      </button>
    ) : null,
}));

const addBlock = vi.fn();
const updateNodeData = vi.fn();
const setBlockMenuOpen = vi.fn();
const addBlockWithPlacement = vi.fn(
  (block: BlockInfo, hardcodedValues?: unknown) => ({
    id: `node-${block.id}`,
    data: { block, metadata: {}, hardcodedValues },
  }),
);

beforeEach(() => {
  addBlock.mockClear();
  updateNodeData.mockClear();
  setBlockMenuOpen.mockClear();
  addBlockWithPlacement.mockClear();
});

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({
    fitView: vi.fn(),
    getViewport: () => ({ x: 0, y: 0, zoom: 1 }),
  }),
  useStoreApi: () => ({
    getState: () => ({ width: 1024, height: 768 }),
  }),
}));

vi.mock("../../../stores/controlPanelStore", () => ({
  useControlPanelStore: (selector: (state: object) => unknown) =>
    selector({ setBlockMenuOpen }),
}));

vi.mock("../../../stores/nodeStore", () => ({
  useNodeStore: (selector?: (state: object) => unknown) => {
    const state = {
      addBlock,
      updateNodeData,
      nodes: [],
    };
    return selector ? selector(state) : state;
  },
}));

vi.mock("./hooks/useAddBlockToBuilder", () => ({
  useAddBlockToBuilder: () => ({ addBlockWithPlacement }),
}));

function disabledBlock(): BlockInfo {
  return {
    id: "disabled-block",
    name: "DisabledBlock",
    description: "Unavailable block",
    inputSchema: {},
    outputSchema: {},
    costs: [],
    categories: [],
    contributors: [],
    staticOutput: false,
    uiType: "standard",
    disabled: true,
    disabledReason: "Missing local configuration",
  } as BlockInfo;
}

function enabledBlock(overrides: Partial<BlockInfo> = {}): BlockInfo {
  return {
    id: "enabled-block",
    name: "EnabledBlock",
    description: "Available block",
    inputSchema: {},
    outputSchema: {},
    costs: [],
    categories: [],
    contributors: [],
    staticOutput: false,
    uiType: "standard",
    disabled: false,
    ...overrides,
  } as BlockInfo;
}

test("disables unavailable blocks and explains why on hover", async () => {
  const user = userEvent.setup();
  render(
    <TooltipProvider>
      <Block blockData={disabledBlock()} title="DisabledBlock" />
    </TooltipProvider>,
  );

  const button = screen.getByRole("button", { name: /disabled/i });
  expect(button).toHaveProperty("disabled", true);

  await user.hover(button.parentElement!);

  expect((await screen.findByRole("tooltip")).textContent).toContain(
    "Missing local configuration",
  );
  expect(addBlockWithPlacement).not.toHaveBeenCalled();
});

test("adds an enabled block on click", async () => {
  const user = userEvent.setup();
  render(
    <TooltipProvider>
      <Block blockData={enabledBlock()} title="EnabledBlock" />
    </TooltipProvider>,
  );

  await user.click(screen.getByRole("button", { name: /enabled/i }));

  expect(addBlockWithPlacement).toHaveBeenCalledWith(
    expect.objectContaining({ id: "enabled-block" }),
  );
  expect(addBlockWithPlacement).toHaveBeenCalledTimes(1);
});

test("opens the MCP tool dialog for MCP blocks", async () => {
  const user = userEvent.setup();
  render(
    <TooltipProvider>
      <Block
        blockData={enabledBlock({ uiType: BlockUIType.MCP_TOOL })}
        title="MCPBlock"
      />
    </TooltipProvider>,
  );

  await user.click(screen.getByRole("button", { name: /mcp/i }));

  expect(addBlockWithPlacement).not.toHaveBeenCalled();
});

test("adds an MCP block with connection details after confirming the dialog", async () => {
  const user = userEvent.setup();
  render(
    <TooltipProvider>
      <Block
        blockData={enabledBlock({ uiType: BlockUIType.MCP_TOOL })}
        title="MCPBlock"
      />
    </TooltipProvider>,
  );

  await user.click(screen.getByRole("button", { name: /mcp/i }));
  await user.click(screen.getByRole("button", { name: /confirm mcp tool/i }));

  expect(addBlockWithPlacement).toHaveBeenCalledTimes(1);
  expect(addBlockWithPlacement).toHaveBeenCalledWith(
    expect.objectContaining({ id: "enabled-block" }),
    expect.objectContaining({
      server_url: "http://localhost:8000",
      server_name: "my-mcp",
      selected_tool: "search",
    }),
  );
});

test("starts a drag on enabled blocks and closes the block menu", async () => {
  const { fireEvent } = await import("@testing-library/react");
  render(
    <TooltipProvider>
      <Block blockData={enabledBlock()} title="EnabledBlock" />
    </TooltipProvider>,
  );

  const button = screen.getByRole("button", { name: /enabled/i });
  const dataTransfer = {
    effectAllowed: "",
    setData: vi.fn(),
    setDragImage: vi.fn(),
  } as unknown as DataTransfer;
  fireEvent.dragStart(button, { dataTransfer });

  expect(setBlockMenuOpen).toHaveBeenCalledWith(false);
});

test("names AGENT blocks with the block's display name on add", async () => {
  const user = userEvent.setup();
  render(
    <TooltipProvider>
      <Block
        blockData={enabledBlock({ id: SpecialBlockID.AGENT, name: "MyAgent" })}
        title="MyAgent"
      />
    </TooltipProvider>,
  );

  await user.click(screen.getByRole("button", { name: /my agent/i }));

  expect(addBlockWithPlacement).toHaveBeenCalledTimes(1);
  expect(updateNodeData).toHaveBeenCalledWith(
    "node-e189baac-8c20-45a1-94a7-55177ea42565",
    expect.objectContaining({
      metadata: expect.objectContaining({ customized_name: "MyAgent" }),
    }),
  );
});
