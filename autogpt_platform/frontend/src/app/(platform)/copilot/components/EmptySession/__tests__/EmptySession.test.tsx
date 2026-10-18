import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  normalizeWhitespace,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EmptySession } from "../EmptySession";

const flags = vi.hoisted(() => ({ experts: true }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "hire-experts" && flags.experts,
    useFlagStatus: (flag: string) => ({
      enabled: flag === "hire-experts" && flags.experts,
      ready: true,
    }),
  };
});

function makeExpert(args: { id: string; name: string; role: string }): Expert {
  return {
    ...args,
    avatar_url: `https://example.com/${args.id}.png`,
    bio: null,
    skills: [],
    tagline: "",
    identity: `You are ${args.name}.`,
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: `template-${args.id}`,
    is_archived: false,
    workflows: [],
  };
}

const mariaExpert = makeExpert({
  id: "expert-maria",
  name: "Maria",
  role: "Marketing Strategist",
});
const maxExpert = makeExpert({ id: "expert-max", name: "Max", role: "" });
const samExpert = makeExpert({ id: "expert-sam", name: "Sam", role: "Sales" });

function renderEmptySession(searchParams: string, onSend = vi.fn()) {
  server.use(
    getListExpertIdentitiesMockHandler([mariaExpert, maxExpert, samExpert]),
  );
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={onSend}
      />
    </Wrapper>,
  );
}

describe("EmptySession — recipient-aware intro", () => {
  it("introduces the selected expert with an inline recipient chip", async () => {
    const { container } = renderEmptySession("?expertId=expert-maria");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "I'm Maria, your Marketing Strategist. What should I take on?",
      ),
    );
    expect(
      screen.getByPlaceholderText("What should Maria work on?"),
    ).toBeDefined();
  });

  it("supports an expert with a short role", async () => {
    const { container } = renderEmptySession("?expertId=expert-sam");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "I'm Sam, your Sales Development Rep. What should I take on?",
      ),
    );
  });

  it("supports an expert without a role", async () => {
    const { container } = renderEmptySession("?expertId=expert-max");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "I'm Max. What should I take on?",
      ),
    );
    expect(
      screen.getByPlaceholderText("What should Max work on?"),
    ).toBeDefined();
  });

  it("keeps the Otto intro without a selected expert", async () => {
    const { container } = renderEmptySession("");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "Tell Otto about your work, and it will find what to automate.",
      ),
    );
    expect(screen.getByPlaceholderText(/What's your role/)).toBeDefined();
  });
});

beforeEach(() => {
  flags.experts = true;
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
  );
});

it("shows the task prompt above the home recap", async () => {
  renderEmptySession("");
  const prompt = screen.getByPlaceholderText(/What's your role/);
  const recap = await screen.findByRole("heading", { name: "Your recap" });
  expect(
    prompt.compareDocumentPosition(recap) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Recent work" })).toBeDefined();
  expect(screen.getByRole("heading", { name: "Your team" })).toBeDefined();
  expect(screen.getByRole("heading", { name: "Now & next" })).toBeDefined();
});

it("keeps the prompt usable when the recap fails to load", async () => {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json({ detail: "failed" }, { status: 500 }),
    ),
  );
  const onSend = vi.fn();
  const user = userEvent.setup();
  renderEmptySession("", onSend);
  await screen.findByText("Your Home briefing could not be loaded");
  await user.type(
    screen.getByPlaceholderText(/What's your role/),
    "Plan my day{Enter}",
  );
  await waitFor(() => expect(onSend).toHaveBeenCalled());
  expect(onSend.mock.calls[0][0]).toBe("Plan my day");
});

it("keeps new tasks available without requesting the recap when experts is off", async () => {
  flags.experts = false;
  const requestedPaths: string[] = [];
  function record({ request }: { request: Request }) {
    requestedPaths.push(new URL(request.url).pathname);
  }
  server.events.on("request:start", record);
  try {
    renderEmptySession("");
    expect(
      await screen.findByPlaceholderText(/What's your role/),
    ).toBeDefined();
    expect(screen.queryByRole("heading", { name: "Your recap" })).toBeNull();
    expect(requestedPaths).not.toContain("/api/proxy/api/home");
  } finally {
    server.events.removeListener("request:start", record);
  }
});

it("shows a named kickoff status and withholds the empty composer", () => {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "" });
  render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={() => {}}
        isKickoffStarting
        expertName="Maria"
      />
    </Wrapper>,
  );
  expect(screen.getByRole("status").textContent).toContain(
    "Opening Maria's workspace",
  );
  expect(screen.queryByPlaceholderText(/What's your role/)).toBeNull();
});

it("switches recipients from the intro with no duplicate picker in the composer", async () => {
  const user = userEvent.setup();
  renderEmptySession("");
  const picker = await screen.findByRole("button", {
    name: "Sending to Otto — change recipient",
  });
  const prompt = screen.getByPlaceholderText(/What's your role/);
  expect(
    picker.compareDocumentPosition(prompt) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(
    screen.getAllByRole("button", { name: /change recipient/, hidden: true }),
  ).toHaveLength(1);
  await user.click(picker);
  await user.click(await screen.findByRole("menuitem", { name: /Maria/ }));
  expect(
    await screen.findByPlaceholderText("What should Maria work on?"),
  ).toBeDefined();
  expect(
    screen.getByRole("button", { name: "Sending to Maria — change recipient" }),
  ).toBeDefined();
});

const teamPlaceholder =
  "What should the team work on? e.g. 'Follow up with yesterday's leads'";

function mockDashboard(dashboard: HomeDashboardResponse) {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(dashboard),
    ),
  );
}

it("keeps the discovery prompt without starter pills or suggestion requests for an empty account", async () => {
  const fetchSuggestions = vi.fn(() => HttpResponse.json({ themes: [] }));
  server.use(http.get("*/api/chat/suggested-prompts", fetchSuggestions));
  renderEmptySession("");
  await screen.findByRole("heading", { name: "Your recap" });
  expect(screen.getByPlaceholderText(/What's your role/)).toBeDefined();
  for (const name of ["Learn", "Create", "Automate", "Organize"]) {
    expect(screen.queryByRole("button", { name })).toBeNull();
  }
  expect(fetchSuggestions).not.toHaveBeenCalled();
});

it.each([
  [
    "an expert hired during onboarding",
    { team: { total: 1, ready: 1, working: 0, needs_attention: 0 } },
  ],
  [
    "running work",
    {
      active_tasks: [
        { id: "run-1", title: "Weekly report", status: "running" },
      ],
    },
  ],
  [
    "scheduled work",
    {
      upcoming_tasks: [
        {
          id: "schedule-1",
          title: "Weekly report",
          kind: "agent",
          next_run_time: new Date(),
        },
      ],
    },
  ],
  [
    "work awaiting attention",
    {
      attention: [
        {
          id: "attention-1",
          kind: "setup",
          priority: "normal",
          title: "Finish setup",
          description: "Connect an account",
          why_it_matters: "Setup is incomplete",
          primary_action: { label: "Open", href: "/team" },
        },
      ],
    },
  ],
  [
    "completed work",
    { week: { ...makeDashboard([]).week, run_count: 1, completed_count: 1 } },
  ],
  ["recent chat output", { recent_work: { total_count: 1, groups: [] } }],
] satisfies [string, Partial<HomeDashboardResponse>][])(
  "uses the team prompt without starter pills for users with %s",
  async (_label, activity) => {
    mockDashboard({ ...makeDashboard([]), ...activity });
    renderEmptySession("");
    expect(await screen.findByPlaceholderText(teamPlaceholder)).toBeDefined();
    for (const name of ["Learn", "Create", "Automate", "Organize"]) {
      expect(screen.queryByRole("button", { name })).toBeNull();
    }
    expect(
      screen.getByRole("button", {
        name: "Sending to Otto — change recipient",
      }),
    ).toBeDefined();
  },
);

it("keeps the chosen expert's prompt for an existing account", async () => {
  mockDashboard({
    ...makeDashboard([]),
    team: { total: 1, ready: 1, working: 0, needs_attention: 0 },
  });
  renderEmptySession("?expertId=expert-maria");
  await screen.findByRole("heading", { name: "Your recap" });
  expect(
    await screen.findByPlaceholderText("What should Maria work on?"),
  ).toBeDefined();
  expect(screen.queryByRole("button", { name: "Learn" })).toBeNull();
});

it("opens a fresh Home visit on Otto after choosing another recipient", async () => {
  const user = userEvent.setup();
  const firstVisit = renderEmptySession("");
  await user.click(
    await screen.findByRole("button", {
      name: "Sending to Otto — change recipient",
    }),
  );
  await user.click(await screen.findByRole("menuitem", { name: /Maria/ }));
  await screen.findByPlaceholderText("What should Maria work on?");
  firstVisit.unmount();

  renderEmptySession("");
  expect(
    await screen.findByRole("button", {
      name: "Sending to Otto — change recipient",
    }),
  ).toBeDefined();
  expect(
    screen.queryByPlaceholderText("What should Maria work on?"),
  ).toBeNull();
});
