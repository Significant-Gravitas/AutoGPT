import { describe, expect, it } from "vitest";
import { getGetV1ListAvailableBlocksMockHandler } from "@/app/api/__generated__/endpoints/blocks/blocks.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { RunBlockTool, type RunBlockToolPart } from "../RunBlock";

const blockId = "87840993-2053-44b7-8da4-187ad4ee518c";

function makePart(overrides: Partial<RunBlockToolPart> = {}): RunBlockToolPart {
  return {
    type: "tool-run_block",
    toolCallId: "call-run-block-1",
    state: "input-available",
    input: { block_id: blockId, input_data: {} },
    ...overrides,
  };
}

describe("RunBlockTool block name lookup", () => {
  it("shows the resolved block name instead of the raw id while running", async () => {
    server.use(
      getGetV1ListAvailableBlocksMockHandler([
        { id: blockId, name: "AIConversationBlock" },
      ]),
    );
    render(<RunBlockTool part={makePart()} />);
    expect(await screen.findByText('Running "AI Conversation"')).not.toBeNull();
  });

  it("shows the raw block id until a name is known", () => {
    server.use(getGetV1ListAvailableBlocksMockHandler([]));
    render(<RunBlockTool part={makePart()} />);
    expect(screen.getByText(`Running "${blockId}"`)).not.toBeNull();
  });

  it("uses block_name from the input without needing the lookup", () => {
    render(
      <RunBlockTool
        part={makePart({
          input: { block_id: blockId, block_name: "My Block", input_data: {} },
        })}
      />,
    );
    expect(screen.getByText('Running "My Block"')).not.toBeNull();
  });
});
