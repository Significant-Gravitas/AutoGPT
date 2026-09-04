import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import {
  ExpertChangeCard,
  ExpertChangeGroup,
  ExpertChangePart,
} from "../ExpertCards";
import type { ChainRow } from "../helpers";
import { ToolResult } from "../ToolResult";

vi.mock("../../../tools/GenericTool/GenericTool", () => ({
  GenericTool: () => <div data-testid="generic-tool" />,
}));

const artifactsFlag = { enabled: false };
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ARTIFACTS: "artifacts" },
  useGetFlag: () => artifactsFlag.enabled,
}));

function row(tool: string, output: unknown): ChainRow {
  return {
    key: tool,
    category: "team",
    text: tool,
    state: "done",
    tool,
    output,
  };
}

describe("expert change cards", () => {
  afterEach(() => {
    artifactsFlag.enabled = false;
    useCopilotUIStore.getState().resetArtifactPanel();
    useCopilotUIStore.getState().setInitialPrompt(null);
  });

  it("shows who the proposed expert is and nothing else", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_proposed",
          message: "Nothing created yet.",
          applied: false,
          confirmation_id: "c-1",
          preview: {
            kind: "raise",
            name: "Otto",
            role: "Inbox triage",
            tagline: "Sorts your morning inbox.",
            about: "You group the morning inbox.",
            boundaries: "You never send a reply yourself.",
            weekly_budget: 2000,
          },
        }}
      />,
    );

    expect(screen.getByText("Raise an expert")).toBeDefined();
    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Inbox triage")).toBeDefined();
    expect(screen.getByText("Sorts your morning inbox.")).toBeDefined();
    // The card is an introduction, not a charter — no status, budget or limits.
    expect(screen.queryByText("You group the morning inbox.")).toBeNull();
    expect(screen.queryByText("Needs your OK")).toBeNull();
    expect(screen.queryByText(/Stops at:/)).toBeNull();
    expect(screen.queryByText(/Weekly budget/)).toBeNull();
  });

  it("opens the full charter in the side panel from Show more", async () => {
    artifactsFlag.enabled = true;
    const user = userEvent.setup();
    render(
      <ExpertChangeCard
        artifactId="call-raise"
        output={{
          type: "expert_change_proposed",
          applied: false,
          preview: {
            kind: "raise",
            name: "Otto",
            role: "Inbox triage",
            tagline: "Sorts your morning inbox.",
            about: "You group the morning inbox.",
            boundaries: "You never send a reply yourself.",
            weekly_budget: 2000,
          },
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Details" }));

    const { activeArtifact, isOpen } =
      useCopilotUIStore.getState().artifactPanel;
    expect(isOpen).toBe(true);
    expect(activeArtifact?.id).toBe("expert:call-raise");
    expect(activeArtifact?.expert).toMatchObject({
      kind: "raise",
      name: "Otto",
      role: "Inbox triage",
      about: "You group the morning inbox.",
      boundaries: "You never send a reply yourself.",
      weeklyBudget: 2000,
      applied: false,
    });
    expect(screen.queryByText("You group the morning inbox.")).toBeNull();
  });

  it("keeps the charter under Show more when the panel is unavailable", async () => {
    const user = userEvent.setup();
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_proposed",
          applied: false,
          preview: {
            kind: "raise",
            name: "Otto",
            tagline: "Sorts your morning inbox.",
            about: "You group the morning inbox.",
            boundaries: "You never send a reply yourself.",
          },
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Details" }));

    expect(screen.getByText("You group the morning inbox.")).toBeDefined();
    expect(
      screen.getByText("Stops at: You never send a reply yourself."),
    ).toBeDefined();
  });

  it("falls back to the charter when an expert has no tagline", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_proposed",
          applied: false,
          preview: { kind: "raise", name: "Otto", about: "You triage." },
        }}
      />,
    );

    expect(screen.getByText("You triage.")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
  });

  it("always offers Details when the panel is available", () => {
    artifactsFlag.enabled = true;
    render(
      <ExpertChangeCard
        artifactId="call-1"
        output={{
          type: "expert_change_proposed",
          applied: false,
          preview: { kind: "raise", name: "Otto", about: "You triage." },
        }}
      />,
    );

    expect(screen.getByRole("button", { name: "Details" })).toBeDefined();
  });

  it("hides Details when the tagline is the whole card", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_proposed",
          applied: false,
          preview: {
            kind: "raise",
            name: "Otto",
            tagline: "Sorts your morning inbox.",
            voice_preferences: "Short sentences.",
            weekly_budget: 2000,
          },
        }}
      />,
    );

    expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
  });

  it("still offers Details without a tagline when there is more to read", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_proposed",
          applied: false,
          preview: {
            kind: "raise",
            name: "Otto",
            about: "You triage.",
            boundaries: "You never reply.",
          },
        }}
      />,
    );

    expect(screen.getByRole("button", { name: "Details" })).toBeDefined();
  });

  it("reads the same once the expert exists", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          message: "Otto is on the team.",
          applied: true,
          kind: "raise",
          expert: {
            id: "exp-otto",
            name: "Otto",
            role: "Inbox triage",
            avatar_url: null,
            color: "violet",
          },
          failed_workflows: ["Inbox sweep"],
        }}
      />,
    );

    expect(screen.getByText("Expert raised")).toBeDefined();
    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Inbox triage")).toBeDefined();
    expect(screen.queryByText("Raised")).toBeNull();
    expect(screen.queryByText(/Couldn't set up/)).toBeNull();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("renders a handoff as the receiving teammate's sub-session", () => {
    render(
      <ToolResult
        row={{
          ...row("handoff_to_expert", {
            type: "mcp_tool_output",
            message: "Bea picked it up.",
            status: "running",
            sub_session_id: "sub-1",
            sub_autopilot_session_link: "/copilot?sessionId=sub-1",
            expert: { id: "exp-b", name: "Bea", role: "Ops lead" },
          }),
          category: "agent",
        }}
      />,
    );

    expect(screen.getByText("Bea")).toBeDefined();
    expect(screen.getByText("Ops lead")).toBeDefined();
    expect(screen.getByLabelText("Open sub-session").getAttribute("href")).toBe(
      "/copilot?sessionId=sub-1",
    );
  });
});

function expertPart(
  state: string,
  output?: unknown,
  id = "call-raise",
): ToolUIPart {
  return {
    type: "tool-raise_expert",
    state,
    toolCallId: id,
    input: {},
    output,
  } as ToolUIPart;
}

function proposal(name: string, id: string): ToolUIPart {
  return expertPart(
    "output-available",
    {
      type: "expert_change_proposed",
      applied: false,
      confirmation_id: `c-${id}`,
      preview: { kind: "raise", name, role: "Engineer" },
    },
    id,
  );
}

describe("ExpertChangePart", () => {
  it("renders the proposal card once the output lands", () => {
    render(
      <ExpertChangePart
        part={proposal("Otto", "a")}
        isCurrentlyStreaming={false}
      />,
    );

    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Engineer")).toBeDefined();
  });

  it("holds the card's shape while the expert is still being written", () => {
    const { container } = render(
      <ExpertChangePart
        part={expertPart("input-available")}
        isCurrentlyStreaming
      />,
    );

    expect(container.querySelectorAll(".animate-pulse").length).toBe(5);
  });

  it("renders nothing for an empty row once the stream is over", () => {
    const { container } = render(
      <ExpertChangePart
        part={expertPart("input-available")}
        isCurrentlyStreaming={false}
      />,
    );

    expect(container.firstChild).toBeNull();
  });

  it("falls back to the generic tool row when the call errored", () => {
    render(
      <ExpertChangePart
        part={
          { ...expertPart("output-error"), errorText: "boom" } as ToolUIPart
        }
        isCurrentlyStreaming={false}
      />,
    );

    expect(screen.getByTestId("generic-tool")).toBeDefined();
  });
});

describe("ExpertChangeGroup", () => {
  it("shows a single expert without a pager", () => {
    render(<ExpertChangeGroup parts={[proposal("Fiona", "a")]} />);

    expect(screen.getByText("Fiona")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Next expert" })).toBeNull();
  });

  it("pages through two or more experts one at a time", async () => {
    const user = userEvent.setup();
    render(
      <ExpertChangeGroup
        parts={[
          proposal("Fiona", "a"),
          proposal("Bhaskar", "b"),
          proposal("Otto", "c"),
        ]}
      />,
    );

    expect(screen.getByText("Fiona")).toBeDefined();
    expect(screen.queryByText("Bhaskar")).toBeNull();
    expect(screen.getByText("1 of 3")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Previous expert" }),
    ).toHaveProperty("disabled", true);

    await user.click(screen.getByRole("button", { name: "Next expert" }));
    expect(screen.getByText("Bhaskar")).toBeDefined();
    expect(screen.getByText("2 of 3")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Go to expert 3" }));
    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByRole("button", { name: "Next expert" })).toHaveProperty(
      "disabled",
      true,
    );

    await user.click(screen.getByRole("button", { name: "Previous expert" }));
    expect(screen.getByText("Bhaskar")).toBeDefined();
  });

  it("leaves empty rows out of the pager once the stream is over", () => {
    const { container } = render(
      <ExpertChangeGroup
        parts={[expertPart("input-available"), expertPart("input-available")]}
        isCurrentlyStreaming={false}
      />,
    );

    expect(container.firstChild).toBeNull();
  });
});

describe("expert approval", () => {
  function prompt(): string | null {
    return useCopilotUIStore.getState().initialPrompt;
  }

  it("moves to the next expert on each decision, then drafts all of them", async () => {
    const user = userEvent.setup();
    render(
      <ExpertChangeGroup
        parts={[proposal("Fiona", "a"), proposal("Bhaskar", "b")]}
      />,
    );

    expect(screen.getByText("Fiona")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Add decisions to message" }),
    ).toBeNull();

    await user.click(screen.getByRole("button", { name: "Approve" }));
    expect(screen.getByText("Bhaskar")).toBeDefined();
    expect(screen.getByText("2 of 2")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Go to expert 1" }).className,
    ).toContain("emerald");

    await user.click(screen.getByRole("button", { name: "Decline" }));
    expect(screen.getByRole("button", { name: "Undo decline" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Go to expert 2" }).className,
    ).toContain("red");
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();

    await user.click(
      screen.getByRole("button", { name: "Add decisions to message" }),
    );
    expect(prompt()).toContain("Approved: create Fiona");
    expect(prompt()).toContain("c-a");
    expect(prompt()).toContain("Not approved: do not create Bhaskar");
    expect(prompt()).toContain("c-b");
    expect(screen.getByText("Added to message")).toBeDefined();
  });

  it("remembers a decision when paging back", async () => {
    const user = userEvent.setup();
    render(
      <ExpertChangeGroup
        parts={[proposal("Fiona", "a"), proposal("Bhaskar", "b")]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(screen.getByRole("button", { name: "Previous expert" }));

    expect(screen.getByText("Fiona")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Unapprove" }));
    expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Go to expert 1" }).className,
    ).not.toContain("emerald");
  });

  it("drafts a single expert straight after its decision", async () => {
    const user = userEvent.setup();
    render(<ExpertChangeGroup parts={[proposal("Otto", "a")]} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(
      screen.getByRole("button", { name: "Add decisions to message" }),
    );

    expect(prompt()).toBe("Approved: create Otto (confirmation_id: c-a).");
  });

  it("drafts the verb the proposal actually asks for", async () => {
    const user = userEvent.setup();
    render(
      <ExpertChangeGroup
        parts={[
          expertPart(
            "output-available",
            {
              type: "expert_change_proposed",
              applied: false,
              confirmation_id: "c-u",
              preview: { kind: "update", name: "Otto", role: "Engineer" },
            },
            "u",
          ),
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(
      screen.getByRole("button", { name: "Add decisions to message" }),
    );

    expect(prompt()).toBe("Approved: update Otto (confirmation_id: c-u).");
  });

  it("offers no decision once applied or on a read-only transcript", () => {
    render(
      <ExpertChangeGroup
        parts={[
          expertPart("output-available", {
            type: "expert_change_applied",
            applied: true,
            kind: "raise",
            expert: { id: "exp-1", name: "Otto", role: "Inbox triage" },
          }),
        ]}
      />,
    );
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    cleanup();

    render(<ExpertChangeGroup parts={[proposal("Otto", "a")]} readOnly />);
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("keeps the pager inside the card, beside the decision", () => {
    render(
      <ExpertChangeGroup
        parts={[proposal("Fiona", "a"), proposal("Bhaskar", "b")]}
      />,
    );

    const card = screen.getByText("Fiona").closest(".rounded-3xl");
    expect(card).not.toBeNull();
    expect(card?.textContent).toContain("1 of 2");
    expect(card?.textContent).toContain("Approve");
  });
});
