import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LearnedNotes } from "../LearnedNotes";

const state = vi.hoisted(() => ({
  isFeatureEnabled: true,
  notes: [] as Array<Record<string, unknown>>,
  isLoading: false,
  isError: false,
  deletingNoteId: null as string | null,
  forgetNote: vi.fn(),
}));

vi.mock("../useLearnedNotes", () => ({
  useLearnedNotes: () => state,
}));

describe("LearnedNotes", () => {
  beforeEach(() => {
    state.isFeatureEnabled = true;
    state.notes = [];
    state.isLoading = false;
    state.isError = false;
    state.deletingNoteId = null;
    state.forgetNote.mockReset();
  });

  it("shows stable learned corrections and allows forgetting one", async () => {
    const user = userEvent.setup();
    state.notes = [
      {
        id: "note-1",
        expert_id: "expert-1",
        text: "Always send drafts before publishing.",
        learned_at: new Date("2026-08-01T00:00:00Z"),
        source_session_id: null,
        source_rule_id: null,
        status: "active",
      },
    ];

    render(
      <LearnedNotes
        expertId="expert-1"
        title={<h2>What I&apos;ve learned</h2>}
      />,
    );

    expect(
      screen.getByText("Always send drafts before publishing."),
    ).toBeDefined();
    await user.click(
      screen.getByRole("button", {
        name: "Forget: Always send drafts before publishing.",
      }),
    );
    expect(state.forgetNote).toHaveBeenCalledWith("note-1");
  });

  it("renders no learned-note section when hire-experts is off", () => {
    state.isFeatureEnabled = false;

    render(
      <LearnedNotes
        expertId="expert-1"
        title={<h2>What I&apos;ve learned</h2>}
      />,
    );

    expect(screen.queryByText("What I've learned")).toBeNull();
  });
});
