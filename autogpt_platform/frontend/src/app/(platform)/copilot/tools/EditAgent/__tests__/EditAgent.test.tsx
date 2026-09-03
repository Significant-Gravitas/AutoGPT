import { render, normalizeWhitespace } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { EditAgentTool, type EditAgentToolPart } from "../EditAgent";

vi.mock(
  "../../../components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: vi.fn(), chatSurface: "copilot" }),
  }),
);

function makePart(
  overrides: Partial<EditAgentToolPart> = {},
): EditAgentToolPart {
  return {
    type: "tool-edit_agent",
    toolCallId: "call-edit-1",
    state: "input-streaming",
    input: {},
    ...overrides,
  };
}

describe("EditAgentTool", () => {
  it("shows the plain loading line (no mini-game) while operating", () => {
    const { container } = render(<EditAgentTool part={makePart()} />);

    expect(normalizeWhitespace(container)).toContain(
      "Editing agent, this might take a minute",
    );
    expect(normalizeWhitespace(container)).not.toContain("Play while you wait");
    expect(normalizeWhitespace(container)).not.toContain("WASD");
  });

  it("renders the agent preview accordion once a preview output arrives", () => {
    const { container } = render(
      <EditAgentTool
        part={makePart({
          state: "output-available",
          output: {
            type: "agent_builder_preview",
            agent_name: "Updated Agent",
            node_count: 1,
            message: "Here is your updated agent",
            agent_json: "{}",
            description: "Now with a new step",
          },
        })}
      />,
    );

    expect(normalizeWhitespace(container)).toContain("Updated Agent");
    expect(normalizeWhitespace(container)).toContain("1 block");
  });
});
