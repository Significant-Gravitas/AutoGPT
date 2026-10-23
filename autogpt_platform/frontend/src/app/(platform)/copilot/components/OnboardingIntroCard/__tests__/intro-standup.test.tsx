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
import { delay, http, HttpResponse } from "msw";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../../EmptySession/EmptySession";

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: unknown[]) => toastSpy(...args),
    useToast: () => ({ ...actual.useToast(), toast: toastSpy }),
  };
});

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
  /** Given how many times the roster has been read so far. */
  setup?: ExpertSetupItem[] | ((rosterReads: number) => ExpertSetupItem[]);
  path?: "A" | "B";
  templatesDelayMs?: number;
  setupDelayMs?: number;
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
    getListExpertTemplatesMockHandler(async () => {
      if (args.templatesDelayMs) await delay(args.templatesDelayMs);
      return TEMPLATES;
    }),
    getListExpertSetupItemsMockHandler(async () => {
      const rosterReads = reads;
      if (args.setupDelayMs) await delay(args.setupDelayMs);
      return typeof args.setup === "function"
        ? args.setup(rosterReads)
        : (args.setup ?? []);
    }),
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

function stubMicrophone(available: boolean) {
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: available ? { getUserMedia: vi.fn() } : undefined,
  });
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
  toastSpy.mockReset();
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
      "Drafts Monday's newsletter from last week's posts",
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
    expect(within(team).getByText("Lead list")).toBeDefined();
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

  it("asks for a connection a hire turns out to need once its install finishes", async () => {
    // A hire's schedules exist only after its install, so the setup gaps do too.
    serve({
      hires: (read) => [
        { ...nadia, setup_status: read === 1 ? "installing" : "ready" },
      ],
      setup: (rosterReads) => (rosterReads > 1 ? [makeSetupItem("nadia")] : []),
    });
    renderHome();

    expect(
      await screen.findByText("Want to start here?", {}, { timeout: 3000 }),
    ).toBeDefined();
    expect(
      await screen.findByRole(
        "button",
        { name: "Connect Notion" },
        { timeout: 5_000 },
      ),
    ).toBeDefined();
  }, 10_000);

  it("holds the ask while a finished install's setup gaps are re-read", async () => {
    // No day-one job, so only the workflows the install brings give a choice.
    const max = makeExpert({
      id: "max",
      name: "Max",
      source_template_id: "tpl-unknown",
    });
    serve({
      hires: (read) => [
        read === 1
          ? { ...max, setup_status: "installing" }
          : {
              ...max,
              workflows: [{ graph_id: "g", name: "Lead list" } as never],
            },
      ],
      setup: (rosterReads) => (rosterReads > 1 ? [makeSetupItem("max")] : []),
      setupDelayMs: 400,
    });
    const asked: string[] = [];
    const observer = new MutationObserver(() => {
      const ask = screen.queryByRole("region", {
        name: "One thing I need from you",
      });
      if (ask?.textContent) asked.push(ask.textContent);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    renderHome();

    await screen.findByRole(
      "button",
      { name: "Connect Notion" },
      { timeout: 6_000 },
    );
    observer.disconnect();
    expect(asked.some((text) => text.includes("Lead list"))).toBe(false);
  }, 10_000);

  it("shows no stale Ready and no job choice when the re-read keeps failing", async () => {
    const max = makeExpert({
      id: "max",
      name: "Max",
      source_template_id: "tpl-unknown",
    });
    serve({
      hires: (read) => [
        read === 1
          ? { ...max, setup_status: "installing" }
          : {
              ...max,
              workflows: [{ graph_id: "g", name: "Lead list" } as never],
            },
      ],
    });
    let setupReads = 0;
    server.use(
      http.get("/api/proxy/api/experts/setup", () => {
        setupReads += 1;
        return setupReads === 1
          ? HttpResponse.json([])
          : HttpResponse.json({ detail: "down" }, { status: 500 });
      }),
    );
    const painted: string[] = [];
    const observer = new MutationObserver(() => {
      const team = screen.queryByRole("region", { name: "Your team" });
      const ask = screen.queryByRole("region", {
        name: "One thing I need from you",
      });
      const status = team
        ? within(team).queryByRole("status")?.textContent
        : undefined;
      painted.push(`${status ?? ""}|${ask?.textContent ?? ""}`);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    renderHome();

    const team = await findTeam();
    await waitFor(
      () =>
        expect(within(team).getByRole("status").textContent).toBe(
          "Couldn't check setup",
        ),
      { timeout: 12_000 },
    );
    observer.disconnect();
    expect(setupReads).toBeGreaterThan(2);
    expect(
      screen.queryByRole("region", { name: "One thing I need from you" }),
    ).toBeNull();
    expect(painted.some((text) => text.startsWith("Ready|"))).toBe(false);
    expect(painted.some((text) => text.includes("Lead list"))).toBe(false);
  }, 20_000);

  it("waits for the day-one jobs before asking, so the choices never change under the cursor", async () => {
    serve({
      hires: [
        {
          ...nadia,
          workflows: [{ graph_id: "g", name: "Newsletter workflow" } as never],
        },
      ],
      templatesDelayMs: 300,
    });
    const asked: string[] = [];
    const observer = new MutationObserver(() => {
      const ask = screen.queryByRole("region", {
        name: "One thing I need from you",
      });
      if (ask?.textContent) asked.push(ask.textContent);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    renderHome();

    await screen.findByRole(
      "button",
      { name: /^Nadia: Drafts Monday's newsletter/ },
      { timeout: 3000 },
    );
    observer.disconnect();
    expect(asked.some((text) => text.includes("Newsletter workflow"))).toBe(
      false,
    );
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

  it("names only the hires whose grant failed", async () => {
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
      http.post("/api/proxy/api/experts/:expertId/credentials", ({ params }) =>
        params.expertId === "robin"
          ? HttpResponse.json({ detail: "nope" }, { status: 500 })
          : HttpResponse.json([]),
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

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't give Robin access to Notion",
        }),
      ),
    );
  });

  it("says so when the pick could not be sent to Otto", async () => {
    serve({ hires: [nadia, robin] });
    renderHome(
      vi.fn(async () => {
        throw new Error("network down");
      }),
    );

    await userEvent.click(
      within(await findAsk()).getByRole("button", { name: /^Robin: Triages/ }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Could not send that to Otto" }),
      ),
    );
  });

  it("offers only typing to a user whose browser cannot record", async () => {
    serve({ hires: [], path: "B" });
    stubMicrophone(false);
    renderHome();

    const ask = await findAsk();
    expect(
      within(ask).queryByRole("button", { name: "Talk me through your week" }),
    ).toBeNull();
    expect(
      within(ask).getByRole("button", { name: "Type it instead" }),
    ).toBeDefined();
  });

  it("asks a user who skipped the dump to talk through their week", async () => {
    serve({ hires: [], path: "B" });
    stubMicrophone(true);
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
