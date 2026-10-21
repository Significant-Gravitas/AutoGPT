import { getGetBrainDumpIntroMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import {
  getListExpertIdentitiesMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../../EmptySession/EmptySession";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", user_metadata: { preferred_name: "Alex" } },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

const GREETING = "You have an invoicing problem and a sales problem.";

function makeExpert(overrides: Partial<Expert> & { id: string; name: string }) {
  return {
    avatar_url: null,
    role: "Sales",
    job_title: null,
    tagline: null,
    bio: null,
    skills: [],
    identity: "",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: `template-${overrides.id}`,
    is_archived: false,
    workflows: [],
    setup_status: "ready",
    ...overrides,
  } satisfies Expert;
}

const maria = makeExpert({
  id: "expert-maria",
  name: "Maria",
  job_title: "Marketing Manager",
});
const max = makeExpert({ id: "expert-max", name: "Max" });

let expertReads = 0;

beforeEach(() => {
  capture.mockReset();
  window.localStorage.clear();
  window.sessionStorage.clear();
  expertReads = 0;
  server.use(
    getGetBrainDumpIntroMockHandler200({
      path: "A",
      greeting: GREETING,
      prompts: [],
      greeting_done: false,
      greeting_pending: false,
      team: null,
      team_pending: false,
    }),
    http.get("*/api/onboarding/brain-dump/suggested-tasks", () =>
      HttpResponse.json({ tasks: [], stored: true }),
    ),
    getListExpertIdentitiesMockHandler([]),
  );
});

function serveExperts(read: (count: number) => Expert[]) {
  server.use(
    getListExpertsMockHandler(() => {
      expertReads += 1;
      return read(expertReads);
    }),
  );
}

function renderGreeting() {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "", hasMemory: true });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={vi.fn()}
      />
    </Wrapper>,
  );
}

function findTeam() {
  return screen.findByRole("region", { name: "Your team" }, { timeout: 3000 });
}

describe("The greeting after onboarding — your team", () => {
  it("lists the hired experts with a Chat link into each one's first chat", async () => {
    serveExperts(() => [
      maria,
      max,
      makeExpert({ id: "template-tom", name: "Tom", is_template: true }),
      makeExpert({ id: "expert-ada", name: "Ada", is_archived: true }),
    ]);
    renderGreeting();

    const team = await findTeam();
    expect(within(team).getByText("Maria")).toBeDefined();
    expect(within(team).getByText("Marketing Manager")).toBeDefined();
    expect(within(team).getByText("Max")).toBeDefined();
    expect(within(team).queryByText("Tom")).toBeNull();
    expect(within(team).queryByText("Ada")).toBeNull();
    expect(
      within(team).getByText("Your team is ready. Open a chat to get started."),
    ).toBeDefined();

    const chat = within(team).getByRole("link", { name: "Chat with Maria" });
    expect(chat.getAttribute("href")).toBe(
      "/home?expertId=expert-maria&kickoff=1",
    );
    fireEvent.click(chat);
    expect(capture).toHaveBeenCalledWith("hired_expert_chat_opened", {
      expert_id: "expert-maria",
      surface: "intro",
      setup_status: "ready",
    });
  });

  it("follows a fresh hire's setup until it is ready", async () => {
    serveExperts((count) => [
      { ...maria, setup_status: count === 1 ? "installing" : "ready" },
    ]);
    renderGreeting();

    const team = await findTeam();
    expect(within(team).getByRole("status").textContent).toBe("Setting up…");
    expect(
      within(team).getByText(/keep talking to me meanwhile/),
    ).toBeDefined();

    await waitFor(
      () => expect(within(team).getByRole("status").textContent).toBe("Ready"),
      { timeout: 5_000 },
    );
    expect(within(team).queryByText(/keep talking to me meanwhile/)).toBeNull();
  });

  it("adds nothing to the greeting when nobody was hired", async () => {
    serveExperts(() => [
      makeExpert({ id: "template-tom", name: "Tom", is_template: true }),
    ]);
    renderGreeting();

    expect(
      await screen.findByText(GREETING, {}, { timeout: 3000 }),
    ).toBeDefined();
    await waitFor(() => expect(expertReads).toBeGreaterThan(0));
    await act(async () => {});
    expect(screen.queryByRole("region", { name: "Your team" })).toBeNull();
  });
});
