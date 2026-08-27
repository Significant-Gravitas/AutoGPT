import {
  getArchiveExpertLearnedNoteMockHandler204,
  getArchiveExpertLearnedNoteMockHandler422,
  getListExpertLearnedNotesMockHandler200,
  getListExpertLearnedNotesMockHandler422,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertLearnedNote } from "@/app/api/__generated__/models/expertLearnedNote";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LearnedNotes } from "../LearnedNotes";

const getFlagMock = vi.hoisted(() => vi.fn(() => true));
const toastMock = vi.hoisted(() => vi.fn());

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: getFlagMock };
});

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

const note: ExpertLearnedNote = {
  id: "note-1",
  expert_id: "expert-1",
  text: "Always send drafts before publishing.",
  learned_at: new Date("2026-08-01T00:00:00Z"),
  source_session_id: null,
  source_rule_id: null,
  status: "active",
};

function renderLearnedNotes(expertId: string | undefined = "expert-1") {
  return render(
    <LearnedNotes
      expertId={expertId}
      title={<h2>What I&apos;ve learned</h2>}
    />,
  );
}

describe("LearnedNotes API integration", () => {
  beforeEach(() => {
    getFlagMock.mockReturnValue(true);
    toastMock.mockReset();
  });

  it("archives a learned note and refreshes the expert's notes", async () => {
    const user = userEvent.setup();
    let archived = false;

    server.use(
      getListExpertLearnedNotesMockHandler200(() => (archived ? [] : [note])),
      getArchiveExpertLearnedNoteMockHandler204(() => {
        archived = true;
      }),
    );

    renderLearnedNotes();

    expect(
      await screen.findByText("Always send drafts before publishing."),
    ).toBeDefined();
    await user.click(
      screen.getByRole("button", {
        name: "Forget: Always send drafts before publishing.",
      }),
    );

    expect(
      await screen.findByText(
        "Nothing recorded yet. What this expert learns will appear here.",
      ),
    ).toBeDefined();
  });

  it("shows a safe error when learned notes cannot be loaded", async () => {
    server.use(getListExpertLearnedNotesMockHandler422());

    renderLearnedNotes();

    expect(
      await screen.findByText("We couldn't load what this expert has learned."),
    ).toBeDefined();
  });

  it("recovers when forgetting a learned note fails", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertLearnedNotesMockHandler200([note]),
      getArchiveExpertLearnedNoteMockHandler422(),
    );

    renderLearnedNotes();

    const forgetButton = await screen.findByRole("button", {
      name: "Forget: Always send drafts before publishing.",
    });
    await user.click(forgetButton);

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith({
        title: "Couldn't forget that note",
        description: "It's still here. Please try again.",
        variant: "destructive",
      }),
    );
    expect((forgetButton as HTMLButtonElement).disabled).toBe(false);
  });

  it("does not load notes when the feature is off or no expert is selected", async () => {
    let requestCount = 0;
    server.use(
      getListExpertLearnedNotesMockHandler200(() => {
        requestCount += 1;
        return [note];
      }),
    );

    getFlagMock.mockReturnValue(false);
    const { rerender } = renderLearnedNotes();
    expect(screen.queryByText("What I've learned")).toBeNull();

    getFlagMock.mockReturnValue(true);
    rerender(
      <LearnedNotes
        expertId={undefined}
        title={<h2>What I&apos;ve learned</h2>}
      />,
    );
    expect(
      await screen.findByText(
        "Nothing recorded yet. What this expert learns will appear here.",
      ),
    ).toBeDefined();
    expect(requestCount).toBe(0);
  });
});
