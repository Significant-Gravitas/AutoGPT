import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ToolChain } from "../ToolChain";
import { ExpertConfirmationContext } from "../ExpertConfirmationContext";

interface Charter {
  id: string;
  name: string;
  role: string;
}

function raisePart({ id, name, role }: Charter): MessagePart {
  return {
    type: "tool-raise_expert",
    state: "output-available",
    toolCallId: `call-${id}`,
    input: { name },
    output: {
      type: "expert_change_proposed",
      message: "Ready for review.",
      applied: false,
      confirmation_id: id,
      preview: {
        kind: "raise",
        name,
        role,
        about: `You own ${role.toLowerCase()}.`,
        boundaries: "You never act externally without approval.",
        weekly_budget: 2000,
      },
    },
  } as MessagePart;
}

const TEAM: Charter[] = [
  { id: "c-1", name: "Otto", role: "Inbox triage" },
  { id: "c-2", name: "Scout", role: "Research" },
  { id: "c-3", name: "Bea", role: "Ops" },
];

function renderTeam(
  onSend: (message: string) => void,
  team = TEAM,
  appliedConfirmationIDs: ReadonlySet<string> = new Set(),
) {
  return render(
    <ExpertConfirmationContext.Provider value={appliedConfirmationIDs}>
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain
          parts={team.map(raisePart)}
          isStreaming={false}
          founderMode
        />
      </CopilotChatActionsProvider>
    </ExpertConfirmationContext.Provider>,
  );
}

describe("TeamPreviewCard", () => {
  afterEach(cleanup);

  it("renders one roster card and one approval", () => {
    renderTeam(vi.fn());

    expect(screen.getByText("3 experts for your team")).toBeDefined();
    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Scout")).toBeDefined();
    expect(screen.getByText("Bea")).toBeDefined();
    expect(
      screen.getAllByRole("button", { name: "Hire selected" }),
    ).toHaveLength(1);
    expect(screen.queryByText("Needs your OK")).toBeNull();
  });

  it("confirms the selected roster in one call", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend);

    await user.click(screen.getByRole("button", { name: "Remove Scout" }));
    await user.click(screen.getByRole("button", { name: "Hire selected" }));

    expect(onSend).toHaveBeenCalledWith(
      "I approve Otto and Bea. Add them to my team in one step.",
    );
  });

  it("keeps removed selections when another proposal streams in", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const { rerender } = render(
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain
          parts={TEAM.slice(0, 2).map(raisePart)}
          isStreaming
          founderMode
        />
      </CopilotChatActionsProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Remove Scout" }));

    rerender(
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain
          parts={TEAM.map(raisePart)}
          isStreaming={false}
          founderMode
        />
      </CopilotChatActionsProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Hire selected" }));

    expect(onSend.mock.calls[0][0]).not.toContain("Mina");
    expect(onSend.mock.calls[0][0]).toContain("Bea");
  });

  it("keeps the single-expert approval card", () => {
    renderTeam(vi.fn(), [TEAM[0]]);

    expect(screen.getByText("Needs your OK")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Hire selected" })).toBeNull();
  });

  it("reconciles an approved roster instead of asking again", () => {
    renderTeam(vi.fn(), TEAM, new Set(TEAM.map((expert) => expert.id)));

    expect(screen.getByText("Team ready")).toBeDefined();
    expect(screen.getAllByText("Hired")).toHaveLength(3);
    expect(screen.queryByRole("button", { name: "Hire selected" })).toBeNull();
  });

  it("confirms only experts that have not already been hired", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend, TEAM, new Set(["c-1"]));

    await user.click(screen.getByRole("button", { name: "Hire selected" }));

    expect(onSend).toHaveBeenCalledWith(
      "I approve Scout and Bea. Add them to my team in one step.",
    );
  });

  it("reconciles a single expert preview", () => {
    renderTeam(vi.fn(), [TEAM[0]], new Set(["c-1"]));

    expect(screen.getByText("Hired")).toBeDefined();
    expect(screen.queryByText("Needs your OK")).toBeNull();
  });
});
