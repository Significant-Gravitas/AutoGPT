import {
  getListExpertLearnedNotesMockHandler,
  getListExpertPodsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2ListLibraryAgentsMockHandler200,
  getGetV2ListLibraryAgentsResponseMock200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertLearnedNote } from "@/app/api/__generated__/models/expertLearnedNote";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const toastMock = vi.hoisted(() => vi.fn());
const learnedNotesFlag = vi.hoisted(() => ({ enabled: true }));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
    useGetFlag: (flag: string) =>
      flag === actual.Flag.EXPERT_LEARNED_NOTES
        ? learnedNotesFlag.enabled
        : actual.useGetFlag(flag as never),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const hiredMaria = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "External actions require approval.",
  ],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
} as unknown as Expert;

// MSW serves the wire shape, where `learned_at` is still a JSON string; orval's
// `useDates` only turns it into a Date on the client side of the boundary.
type LearnedNoteWire = Omit<ExpertLearnedNote, "learned_at"> & {
  learned_at: string;
};

function makeNote(over: Partial<LearnedNoteWire> = {}): ExpertLearnedNote {
  return {
    id: "note-1",
    expert_id: "expert-maria",
    text: "Always send drafts before publishing.",
    learned_at: "2026-08-01T09:00:00Z",
    source_session_id: null,
    source_rule_id: "edge-1",
    status: "active",
    ...over,
  } as unknown as ExpertLearnedNote;
}

beforeEach(() => {
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(emptyLibraryResponse()),
    getListExpertsMockHandler([hiredMaria]),
  );
});

afterEach(() => {
  learnedNotesFlag.enabled = true;
  toastMock.mockReset();
});

function emptyLibraryResponse() {
  const base = getGetV2ListLibraryAgentsResponseMock200();
  return { ...base, agents: [] };
}

// The archive endpoint answers 204, so it is stubbed by hand rather than
// through the generated handler — there is no response body to mock.
function archiveHandler(onRequest: (path: string) => Response) {
  return http.delete(
    "*/api/experts/:expertId/learned-notes/:noteId",
    ({ request }) => onRequest(new URL(request.url).pathname),
  );
}

async function openSoulDrawer(user: ReturnType<typeof userEvent.setup>) {
  render(<TeamPage />);
  await user.click(await screen.findByRole("button", { name: "Edit Soul" }));
}

describe("SoulDrawer — what I've learned", () => {
  test("lists the expert's learned notes with when they were learned", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertLearnedNotesMockHandler([
        makeNote(),
        makeNote({
          id: "note-2",
          text: "Never email the client on weekends.",
          learned_at: "2026-08-04T09:00:00Z",
        }),
      ]),
    );

    await openSoulDrawer(user);

    expect(
      await screen.findByText("Always send drafts before publishing."),
    ).toBeDefined();
    expect(
      screen.getByText("Never email the client on weekends."),
    ).toBeDefined();
    expect(screen.getAllByText(/^learned /)).toHaveLength(2);
    expect(screen.queryByText(/Nothing recorded yet/)).toBeNull();
  });

  test("keeps the empty state when the expert has learned nothing", async () => {
    const user = userEvent.setup();
    server.use(getListExpertLearnedNotesMockHandler([]));

    await openSoulDrawer(user);

    expect(await screen.findByText(/Nothing recorded yet/)).toBeDefined();
  });

  test("archives a note and refetches the list when it is forgotten", async () => {
    const user = userEvent.setup();
    let listRequests = 0;
    let archivedPath = "";
    server.use(
      getListExpertLearnedNotesMockHandler(() => {
        listRequests += 1;
        return listRequests === 1 ? [makeNote()] : [];
      }),
      archiveHandler((path) => {
        archivedPath = path;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    await openSoulDrawer(user);

    await user.click(
      await screen.findByRole("button", {
        name: "Forget: Always send drafts before publishing.",
      }),
    );

    await waitFor(() => expect(listRequests).toBeGreaterThan(1));
    expect(archivedPath).toContain(
      "/api/experts/expert-maria/learned-notes/note-1",
    );
    expect(await screen.findByText(/Nothing recorded yet/)).toBeDefined();
  });

  test("keeps the note and explains when forgetting fails", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertLearnedNotesMockHandler([makeNote()]),
      archiveHandler(() => new HttpResponse(null, { status: 404 })),
    );

    await openSoulDrawer(user);

    await user.click(
      await screen.findByRole("button", {
        name: "Forget: Always send drafts before publishing.",
      }),
    );

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't forget that note",
          variant: "destructive",
        }),
      ),
    );
    expect(
      screen.getByText("Always send drafts before publishing."),
    ).toBeDefined();
  });

  test("hides the section entirely when the flag is off", async () => {
    const user = userEvent.setup();
    learnedNotesFlag.enabled = false;
    server.use(getListExpertLearnedNotesMockHandler([makeNote()]));

    await openSoulDrawer(user);

    await screen.findByRole("dialog", { name: "Maria's Soul" });
    expect(screen.queryByText("What I've learned")).toBeNull();
    expect(
      screen.queryByText("Always send drafts before publishing."),
    ).toBeNull();
  });
});
