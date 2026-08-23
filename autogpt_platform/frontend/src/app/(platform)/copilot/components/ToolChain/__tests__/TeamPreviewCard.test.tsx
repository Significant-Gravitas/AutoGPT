import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ToolChain } from "../ToolChain";

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
      message: "Nothing created yet.",
      applied: false,
      confirmation_id: id,
      preview: {
        kind: "raise",
        name,
        role,
        about: `You own ${role.toLowerCase()}.`,
        boundaries: "You never send a reply yourself.",
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

function renderTeam(onSend: (message: string) => void, team: Charter[] = TEAM) {
  return render(
    <CopilotChatActionsProvider onSend={onSend}>
      <ToolChain parts={team.map(raisePart)} isStreaming={false} />
    </CopilotChatActionsProvider>,
  );
}

describe("TeamPreviewCard", () => {
  afterEach(cleanup);

  it("gathers a turn's proposals into one roster with a single confirm", () => {
    renderTeam(vi.fn());

    expect(screen.getByText("3 experts for your team")).toBeDefined();
    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Scout")).toBeDefined();
    expect(screen.getByText("Bea")).toBeDefined();
    expect(screen.getByText("Inbox triage")).toBeDefined();
    expect(screen.getAllByRole("button", { name: "Hire all" })).toHaveLength(1);
    // The per-expert approval card is replaced, not stacked underneath.
    expect(screen.queryByText("Needs your OK")).toBeNull();
  });

  it("confirms every proposal in one batched call", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend);

    await user.click(screen.getByRole("button", { name: "Hire all" }));

    expect(onSend).toHaveBeenCalledWith(
      'I approve Otto, Scout and Bea. Call confirm_expert_change once with confirmation_ids ["c-1", "c-2", "c-3"].',
    );
  });

  it("leaves a removed expert out of the confirm", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend);

    await user.click(screen.getByRole("button", { name: "Remove Scout" }));

    expect(screen.getByText("2 of 3 selected")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Hire all" }));

    expect(onSend).toHaveBeenCalledWith(
      'I approve Otto and Bea. Call confirm_expert_change once with confirmation_ids ["c-1", "c-3"].',
    );
  });

  it("puts a removed expert back", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend);

    await user.click(screen.getByRole("button", { name: "Remove Bea" }));
    await user.click(screen.getByRole("button", { name: "Put Bea back" }));
    await user.click(screen.getByRole("button", { name: "Hire all" }));

    expect(onSend).toHaveBeenCalledWith(
      'I approve Otto, Scout and Bea. Call confirm_expert_change once with confirmation_ids ["c-1", "c-2", "c-3"].',
    );
  });

  it("falls back to the single-id parameter when one expert is left", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend);

    await user.click(screen.getByRole("button", { name: "Remove Scout" }));
    await user.click(screen.getByRole("button", { name: "Remove Bea" }));
    await user.click(screen.getByRole("button", { name: "Hire all" }));

    expect(onSend).toHaveBeenCalledWith(
      'I approve Otto. Call confirm_expert_change with confirmation_id "c-1".',
    );
  });

  it("cannot confirm an empty roster", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderTeam(onSend);

    for (const member of TEAM) {
      await user.click(
        screen.getByRole("button", { name: `Remove ${member.name}` }),
      );
    }

    expect(
      screen.getByRole("button", { name: "Hire all" }).hasAttribute("disabled"),
    ).toBe(true);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("opens one charter at a time", async () => {
    const user = userEvent.setup();
    renderTeam(vi.fn());

    const otto = screen.getByRole("button", { name: "Otto Inbox triage" });
    const scout = screen.getByRole("button", { name: "Scout Research" });
    expect(otto.getAttribute("aria-expanded")).toBe("false");

    await user.click(otto);
    expect(otto.getAttribute("aria-expanded")).toBe("true");

    await user.click(scout);
    expect(otto.getAttribute("aria-expanded")).toBe("false");
    expect(scout.getAttribute("aria-expanded")).toBe("true");
  });

  it("names each draft in the chain without repeating the roster", async () => {
    const user = userEvent.setup();
    renderTeam(vi.fn());

    await user.click(
      screen.getByRole("button", { name: /review the new team/i }),
    );

    // Otto's row is the carrier — it holds the roster and reads "Review the
    // new team"; every other draft keeps its own name in the chain.
    expect(screen.getAllByText("Drafted Scout").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Drafted Bea").length).toBeGreaterThan(0);
    expect(screen.queryByText("Drafted Otto")).toBeNull();
    expect(screen.getAllByRole("button", { name: "Hire all" })).toHaveLength(1);
  });

  it("keeps a removal when a later proposal streams in", async () => {
    // Regression: the roster used to hang off the LAST proposal row, so each
    // new proposal moved the card to a different row, remounting it and
    // silently restoring experts the user had already removed.
    const user = userEvent.setup();
    const onSend = vi.fn();
    const { rerender } = render(
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain parts={TEAM.slice(0, 2).map(raisePart)} isStreaming />
      </CopilotChatActionsProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Remove Scout" }));

    rerender(
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain parts={TEAM.map(raisePart)} isStreaming={false} />
      </CopilotChatActionsProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Hire all" }));

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend.mock.calls[0][0]).not.toContain("c-2");
    expect(onSend.mock.calls[0][0]).toContain("c-3");
  });

  it("keeps the single-expert card when only one is proposed", () => {
    renderTeam(vi.fn(), [TEAM[0]]);

    expect(screen.getByText("Needs your OK")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Hire all" })).toBeNull();
  });
});
