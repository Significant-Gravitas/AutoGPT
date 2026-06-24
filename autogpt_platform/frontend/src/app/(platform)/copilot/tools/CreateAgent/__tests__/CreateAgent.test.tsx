import { render, normalizeWhitespace } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { CreateAgentTool, type CreateAgentToolPart } from "../CreateAgent";

vi.mock(
  "../../../components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: vi.fn(), chatSurface: "copilot" }),
  }),
);

function makePart(
  overrides: Partial<CreateAgentToolPart> = {},
): CreateAgentToolPart {
  return {
    type: "tool-create_agent",
    toolCallId: "call-create-1",
    state: "input-streaming",
    input: {},
    ...overrides,
  };
}

describe("CreateAgentTool", () => {
  it("shows the plain loading line (no mini-game) while operating", () => {
    const { container } = render(<CreateAgentTool part={makePart()} />);

    expect(normalizeWhitespace(container)).toContain(
      "Creating agent, this might take a minute",
    );
    expect(normalizeWhitespace(container)).not.toContain("Play while you wait");
    expect(normalizeWhitespace(container)).not.toContain("WASD");
  });

  it("renders the agent preview accordion once a preview output arrives", () => {
    const { container } = render(
      <CreateAgentTool
        part={makePart({
          state: "output-available",
          output: {
            type: "agent_builder_preview",
            agent_name: "News Summarizer",
            node_count: 3,
            message: "Here is your agent",
            agent_json: "{}",
            description: "Summarizes the news",
          },
        })}
      />,
    );

    expect(normalizeWhitespace(container)).toContain("News Summarizer");
    expect(normalizeWhitespace(container)).toContain("3 blocks");
  });

  it("renders the suggested-goal card when the goal needs refinement", () => {
    const { container } = render(
      <CreateAgentTool
        part={makePart({
          state: "output-available",
          output: {
            type: "suggested_goal",
            message: "Your goal is a bit vague",
            suggested_goal: "Summarize tech news every morning",
            reason: "More specific",
            goal_type: "vague",
          },
        })}
      />,
    );

    expect(normalizeWhitespace(container)).toContain("Goal needs refinement");
  });
});
