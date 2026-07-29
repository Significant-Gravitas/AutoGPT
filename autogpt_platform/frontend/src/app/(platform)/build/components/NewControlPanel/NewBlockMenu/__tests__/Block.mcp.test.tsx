import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { ReactFlowProvider } from "@xyflow/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { BlockUIType } from "@/lib/autogpt-server-api";
import { useNodeStore } from "../../../../stores/nodeStore";
import { Block } from "../Block";

vi.mock("@/lib/oauth-popup", () => ({ openOAuthPopup: vi.fn() }));

const mockDiscover = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2DiscoverAvailableToolsOnAnMcpServer: (...args: unknown[]) =>
    mockDiscover(...args),
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
  postV2StoreABearerTokenForAnMcpServer: vi.fn(),
}));

const SERVER_URL = "https://mcp.datafa.st/mcp";

const mcpBlock = {
  id: "a0a4b1c2-d3e4-4f56-a7b8-c9d0e1f2a3b4",
  name: "MCPToolBlock",
  description: "Connect to any MCP server",
  categories: [],
  inputSchema: { type: "object", properties: {}, required: [] },
  outputSchema: { type: "object", properties: {} },
  uiType: BlockUIType.MCP_TOOL,
  costs: [],
} as unknown as BlockInfo;

describe("Block — adding an MCP tool block", () => {
  afterEach(() => {
    cleanup();
    mockDiscover.mockReset();
    useNodeStore.setState({ nodes: [] });
  });

  async function addMCPBlock() {
    mockDiscover.mockResolvedValue({
      status: 200,
      data: {
        tools: [
          {
            name: "get_analytics",
            description: "Fetch analytics",
            input_schema: { type: "object", properties: {}, required: [] },
          },
        ],
        server_name: "datafa.st",
      },
    });

    render(
      <ReactFlowProvider>
        <Block blockData={mcpBlock} title="MCP" />
      </ReactFlowProvider>,
    );

    fireEvent.click(screen.getByText("MCP"));
    fireEvent.change(await screen.findByLabelText(/server url/i), {
      target: { value: SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover tools/i }));
    fireEvent.click(await screen.findByText("get_analytics"));
    fireEvent.click(screen.getByRole("button", { name: /add block/i }));

    return useNodeStore.getState().nodes.at(-1);
  }

  it("does not mark credentials as optional", async () => {
    const node = await addMCPBlock();

    // `credentials_optional` means "skip this block if credentials are not
    // configured" to the executor. MCP servers that need no auth would then
    // never run — the graph would finish immediately with zero nodes.
    expect(node?.data.metadata?.credentials_optional).toBeUndefined();
  });

  it("names the node after the server and selected tool", async () => {
    const node = await addMCPBlock();

    expect(node?.data.metadata?.customized_name).toBe(
      "datafa.st: Get Analytics",
    );
  });
});
