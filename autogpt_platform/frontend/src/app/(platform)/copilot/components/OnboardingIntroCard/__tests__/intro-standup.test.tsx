import { getGetBrainDumpIntroMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import {
  getListExpertIdentitiesMockHandler,
  getListExpertSetupItemsMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListProvidersMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertSetupItem } from "@/app/api/__generated__/models/expertSetupItem";
import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { server } from "@/mocks/mock-server";
import { useComposerVoiceRequest } from "@/services/onboarding/composer-voice-request";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../../EmptySession/EmptySession";

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
    source_template_id: `tpl-${overrides.id}`,
    is_archived: false,
    workflows: [],
    setup_status: "ready",
    ...overrides,
  } satisfies Expert;
}

function makeTemplate(id: string, dayOne: string[]): ExpertTemplate {
  return {
    ...makeExpert({
      id,
      name: id,
      is_template: true,
      source_template_id: null,
    }),
    day_one: dayOne.map((title) => ({ title })),
  };
}

function makeSetupItem(
  expertId: string,
  overrides: Partial<ExpertSetupItem> = {},
): ExpertSetupItem {
  return {
    expert_id: expertId,
    expert_name: expertId,
    expert_avatar_url: null,
    workflow_id: `wf-${expertId}`,
    workflow_name: "Weekly digest",
    library_agent_id: null,
    providers: ["notion"],
    resolution: "connect",
    credential_id: null,
    ...overrides,
  };
}

const nadia = makeExpert({
  id: "nadia",
  name: "Nadia",
  job_title: "Content Marketer",
});
const robin = makeExpert({ id: "robin", name: "Robin" });
const frankie = makeExpert({ id: "frankie", name: "Frankie" });
const TEMPLATES = [
  makeTemplate("tpl-nadia", [
    "Drafts Monday's newsletter from last week's posts",
  ]),
  makeTemplate("tpl-robin", ["Triages the support inbox each morning"]),
  makeTemplate("tpl-frankie", ["Sends you a booking link and a daily agenda"]),
];

function serve(args: {
  hires: Expert[] | ((read: number) => Expert[]);
  setup?: ExpertSetupItem[];
  path?: "A" | "B";
}) {
  let reads = 0;
  server.use(
    getGetBrainDumpIntroMockHandler200({
      path: args.path ?? "A",
      greeting: "You have a newsletter problem and a support problem.",
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
    getListExpertsMockHandler(() => {
      reads += 1;
      return typeof args.hires === "function" ? args.hires(reads) : args.hires;
    }),
    getListExpertTemplatesMockHandler(TEMPLATES),
    getListExpertSetupItemsMockHandler(args.setup ?? []),
    getGetV1ListProvidersMockHandler([
      { name: "notion", supported_auth_types: ["api_key"] },
    ]),
  );
}

function renderHome(onStartTask = vi.fn(async () => "session-1")) {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "", hasMemory: true });
  render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={vi.fn()}
        onStartTask={onStartTask}
      />
    </Wrapper>,
  );
  return onStartTask;
}

function findTeam() {
  return screen.findByRole("region", { name: "Your team" }, { timeout: 3000 });
}

function findAsk() {
  return screen.findByRole(
    "region",
    { name: "One thing I need from you" },
    { timeout: 3000 },
  );
}

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("Otto's first turn on Home — the team", () => {
  it("lists the hires with their day-one job and status, three at a time", async () => {
    serve({
      hires: [
        nadia,
        { ...robin, setup_status: "installing" },
        { ...frankie, last_run_status: "RUNNING" },
        makeExpert({
          id: "max",
          name: "Max",
          source_template_id: "tpl-unknown",
          workflows: [{ graph_id: "g", name: "Lead list" } as never],
        }),
        makeExpert({ id: "tom", name: "Tom", is_archived: true }),
      ],
    });
    renderHome();

    const team = await findTeam();
    const rows = within(team).getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    expect(rows[0].textContent).toContain("Content Marketer");
    expect(rows[0].textContent).toContain(
      "Day one: Drafts Monday's newsletter from last week's posts",
    );
    expect(within(rows[0]).getByRole("status").textContent).toBe("Ready");
    expect(within(rows[1]).getByRole("status").textContent).toBe("Setting up");
    expect(within(rows[2]).getByRole("status").textContent).toBe("Working");
    expect(
      within(team).getByRole("link", { name: "and 1 more" }),
    ).toBeDefined();
    expect(within(team).queryByText("Tom")).toBeNull();
    expect(within(team).queryByRole("link", { name: /Chat/ })).toBeNull();
  });

  it("falls back to the first installed workflow when the template promises nothing", async () => {
    serve({
      hires: [
        makeExpert({
          id: "max",
          name: "Max",
          source_template_id: "tpl-unknown",
          workflows: [{ graph_id: "g", name: "Lead list" } as never],
        }),
      ],
    });
    renderHome();

    const team = await findTeam();
    expect(within(team).getByText("Day one: Lead list")).toBeDefined();
  });

  it("follows a hire's setup until it is ready", async () => {
    serve({
      hires: (read) => [
        { ...nadia, setup_status: read === 1 ? "installing" : "ready" },
      ],
    });
    renderHome();

    const team = await findTeam();
    expect(within(team).getByRole("status").textContent).toBe("Setting up");
    await waitFor(
      () => expect(within(team).getByRole("status").textContent).toBe("Ready"),
      { timeout: 5_000 },
    );
  });
});

describe("Otto's first turn on Home — the one ask", () => {
  it("asks for the connection the most hires are waiting on, and Not now moves on", async () => {
    serve({
      hires: [nadia, robin, frankie],
      setup: [makeSetupItem("nadia"), makeSetupItem("robin")],
    });
    renderHome();

    const ask = await findAsk();
    expect(ask.textContent).toContain(
      "Nadia and Robin need Notion for their scheduled work.",
    );
    const team = await findTeam();
    expect(
      within(within(team).getAllByRole("listitem")[0]).getByRole("status")
        .textContent,
    ).toBe("Needs a connection");

    fireEvent.click(within(ask).getByRole("button", { name: "Not now" }));
    expect(
      await screen.findByText("Which should the team start on first?"),
    ).toBeDefined();
  });

  it("gives every waiting hire the connection once it is made", async () => {
    const granted: string[] = [];
    serve({
      hires: [nadia, robin, frankie],
      setup: [makeSetupItem("nadia"), makeSetupItem("robin")],
    });
    server.use(
      http.post("/api/proxy/api/integrations/notion/credentials", () =>
        HttpResponse.json(
          {
            id: "cred-notion",
            provider: "notion",
            type: "api_key",
            title: "N",
          },
          { status: 201 },
        ),
      ),
      http.post(
        "/api/proxy/api/experts/:expertId/credentials",
        ({ params }) => {
          granted.push(String(params.expertId));
          return HttpResponse.json([]);
        },
      ),
    );
    renderHome();

    fireEvent.click(
      within(await findAsk()).getByRole("button", { name: "Connect Notion" }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      await within(dialog).findByRole("button", { name: /API Key/ }),
    );
    await userEvent.type(
      await within(dialog).findByPlaceholderText("My Notion key"),
      "N",
    );
    await userEvent.type(within(dialog).getByPlaceholderText("sk-..."), "x");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Continue" }),
    );

    await waitFor(() => expect(granted.sort()).toEqual(["nadia", "robin"]));
  });

  it("asks which job starts first, and sends the pick to Otto as a message", async () => {
    serve({ hires: [nadia, robin] });
    const onStartTask = renderHome();

    const ask = await findAsk();
    expect(ask.textContent).toContain("Which should the team start on first?");
    await userEvent.click(
      within(ask).getByRole("button", { name: /^Robin: Triages/ }),
    );

    expect(onStartTask).toHaveBeenCalledWith(
      "Let's start with Robin's day-one job: Triages the support inbox each morning.",
    );
  });

  it("asks a user who skipped the dump to talk through their week", async () => {
    serve({ hires: [], path: "B" });
    renderHome();

    const ask = await findAsk();
    const before = useComposerVoiceRequest.getState().count;
    fireEvent.click(
      within(ask).getByRole("button", { name: "Talk me through your week" }),
    );
    expect(useComposerVoiceRequest.getState().count).toBe(before + 1);
    expect(screen.queryByRole("region", { name: "Your team" })).toBeNull();
  });
});
