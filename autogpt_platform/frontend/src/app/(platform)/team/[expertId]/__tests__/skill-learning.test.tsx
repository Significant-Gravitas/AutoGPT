import { Toaster } from "@/components/molecules/Toast/toaster";
import {
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetHomeDashboardMockHandler,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import {
  getEditLearnedSkillMockHandler200,
  getUpdateSkillLearningPolicyMockHandler200,
  getGetSkillLearningDetailMockHandler200,
  getListSkillLearningHistoryMockHandler200,
  getRestoreLearnedSkillVersionMockHandler200,
  getUpdateExpertLearningPolicyMockHandler200,
} from "@/app/api/__generated__/endpoints/skill-learning/skill-learning.msw";
import { SkillUseSummary } from "@/app/api/__generated__/models/skillUseSummary";
import { Expert } from "@/app/api/__generated__/models/expert";
import { SkillLearningDetail } from "@/app/api/__generated__/models/skillLearningDetail";
import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
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
      flag === "hire-experts" || flag === "dream-skill-learning-enabled",
  };
});

const searchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team/expert-1",
  useSearchParams: () => searchParams,
  useParams: () => ({ expertId: "expert-1" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const expert: Expert = {
  id: "expert-1",
  name: "Alex",
  avatar_url: null,
  role: "Data Analyst",
  bio: null,
  skills: ["csv-import-checks"],
  tagline: null,
  identity: "You validate imports.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  learning_paused_at: null,
};

function version(
  overrides: Partial<SkillVersionSummary> & { id: string; version: number },
): SkillVersionSummary {
  return {
    skill_name: "csv-import-checks",
    expert_id: "expert-1",
    origin: "saved_overnight",
    origin_label: "Saved overnight",
    summary: "Added an encoding check after the previous import failed.",
    state: "ready",
    state_label: "Ready to use",
    state_reason: "",
    created_at: new Date("2026-09-14T03:00:00Z"),
    description: "Import a CSV with an encoding check",
    triggers: ["csv import"],
    body: "---\nname: csv-import-checks\n---\n\n## Steps\n1. Open as utf-8\n2. Validate rows\n",
    evidence: [
      { kind: "outcome", ref: "", label: "Worked once in the source" },
    ],
    limits: ["Only the sample fixture was checked."],
    sources: [
      {
        source_id: "src-1",
        source_kind: "chat_session",
        revision: "000000000003",
        accessible: true,
        source_ref: "session-1",
        title: "Fix the CSV import",
        url: "/copilot?sessionId=session-1&expertId=expert-1",
        excluded: false,
      },
      {
        source_id: "src-2",
        source_kind: "chat_session",
        revision: "000000000009",
        accessible: false,
        source_ref: null,
        title: null,
        url: null,
        excluded: false,
      },
    ],
    use: {
      version_id: overrides.id,
      loads: 1,
      checks_passed: 0,
      checks_failed: 0,
      reported_working: 0,
      reported_failed: 0,
      stopped_mid_use: 0,
      unknown: 0,
      reuse_label: "Loaded 1 time · Outcome unknown",
    },
    base_version_id: null,
    restored_from_version_id: null,
    blocked_pattern_class: null,
    blocked_step: null,
    ...overrides,
  };
}

const v1 = version({
  id: "ver-1",
  version: 1,
  origin: "saved_during_work",
  origin_label: "Saved during work",
});
const v2 = version({
  id: "ver-2",
  version: 2,
  base_version_id: "ver-1",
  body: "## Steps\n1. Open as utf-8\n2. Validate rows\n3. Report the count\n",
});

const detail: SkillLearningDetail = {
  skill_name: "csv-import-checks",
  expert_id: "expert-1",
  state: "ready",
  state_label: "Ready to use",
  current_version: v2,
  open_decision: null,
  versions: [v2, v1],
  policy: { auto_improve: true, learning_paused: false, use_paused: false },
  use: v2.use as SkillUseSummary,
};

beforeEach(() => {
  searchParams.delete("tab");
  searchParams.delete("skill");
  searchParams.delete("version");
  server.use(
    getGetHomeDashboardMockHandler(
      getGetHomeDashboardResponseMock200({ attention: [] }),
    ),
    getGetExpertMockHandler(expert),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({ timezone: "UTC", days: [] }),
    getListCopilotSkillsMockHandler200([
      {
        name: "csv-import-checks",
        description: "Import a CSV with an encoding check",
        triggers: ["csv import"],
      },
    ]),
    getListSkillLearningHistoryMockHandler200({
      expert_id: "expert-1",
      items: [
        {
          id: "ver-2",
          kind: "version",
          skill_name: "csv-import-checks",
          expert_id: "expert-1",
          created_at: new Date("2026-09-14T03:00:00Z"),
          summary: "Added an encoding check after the previous import failed.",
          origin: "saved_overnight",
          origin_label: "Saved overnight",
          state: "ready",
          state_label: "Ready to use",
          version: 2,
          version_id: "ver-2",
          source: null,
        },
      ],
    }),
    getGetSkillLearningDetailMockHandler200(detail),
  );
});

afterEach(() => {
  window.localStorage.removeItem("team-workflows-view");
});

async function openSkillsTab() {
  await userEvent.click(await screen.findByRole("tab", { name: "Skills" }));
}

describe("Expert skills — learning", () => {
  test("shows the recent overnight change and a per-skill learning line", async () => {
    render(<ExpertDetailPage />);
    await openSkillsTab();

    const summary = await screen.findByTestId("expert-learning-summary");
    expect(
      within(summary).getByText(
        /Added an encoding check after the previous import failed\. · Saved overnight · Ready to use/,
      ),
    ).toBeDefined();
    expect(
      (await screen.findByTestId("skill-learning-line")).textContent,
    ).toContain("v2 · Saved overnight · Ready to use");
  });

  test("opens the detail sheet from the keyboard, hides inaccessible evidence, and restores a prior version", async () => {
    const restore = vi.fn();
    server.use(
      getRestoreLearnedSkillVersionMockHandler200(
        async ({ request, params }) => {
          restore({ params, body: await request.json() });
          return {
            status: "applied",
            status_label: "Applied",
            reason: "Restored by the owner (from v1)",
            version: null,
            pattern_class: null,
            blocked_step: null,
          };
        },
      ),
    );
    render(<ExpertDetailPage />);
    await openSkillsTab();

    const details = await screen.findByRole("button", {
      name: "Learning details for csv-import-checks",
    });
    details.focus();
    await userEvent.keyboard("{Enter}");

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(within(panel).getByText("Worked once in the source")).toBeDefined();
    expect(within(panel).getByTestId("reuse-label").textContent).toContain(
      "Loaded 1 time · Outcome unknown",
    );

    await userEvent.click(within(panel).getByRole("tab", { name: "Sources" }));
    expect(await within(panel).findByText("Fix the CSV import")).toBeDefined();
    expect(
      within(panel).getByText("Evidence not visible to you"),
    ).toBeDefined();

    await userEvent.click(within(panel).getByRole("tab", { name: "History" }));
    await userEvent.click(
      await within(panel).findByRole("button", { name: "Restore v1" }),
    );
    await waitFor(() => expect(restore).toHaveBeenCalledTimes(1));
    expect(restore.mock.calls[0][0]).toMatchObject({
      params: { name: "csv-import-checks" },
      body: { version_id: "ver-1" },
    });
  });

  test("pauses learning for the expert without touching the skill list", async () => {
    const policy = vi.fn();
    server.use(
      getUpdateExpertLearningPolicyMockHandler200(
        async ({ request }: { request: Request }) => {
          policy(await request.json());
          return { expert_id: "expert-1", learning_paused_at: new Date() };
        },
      ),
    );
    render(<ExpertDetailPage />);
    await openSkillsTab();

    await userEvent.click(
      await screen.findByRole("switch", { name: "Nightly learning for Alex" }),
    );
    await waitFor(() =>
      expect(policy).toHaveBeenCalledWith({ learning_paused: true }),
    );
    expect(screen.getAllByTestId("expert-skill-row")).toHaveLength(1);
  });

  test("deep links from the chat chip open the requested skill", async () => {
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    render(<ExpertDetailPage />);

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(await within(panel).findByTestId("reuse-label")).toBeDefined();
    expect(within(panel).getAllByText("Ready to use").length).toBeGreaterThan(
      0,
    );
  });

  test("keeps the edit draft when the server answers 200 with a domain rejection", async () => {
    server.use(
      getEditLearnedSkillMockHandler200(() => ({
        status: "blocked_content",
        status_label: "Blocked by content check",
        reason:
          "Blocked by content check: github_token in SKILL.md at section 1 › step 1 › line 2",
        version: null,
        pattern_class: "github_token",
        blocked_step: "section 1 › step 1 › line 2",
      })),
    );
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    render(
      <>
        <ExpertDetailPage />
        <Toaster />
      </>,
    );

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    await userEvent.click(
      await within(panel).findByRole("button", { name: "Edit skill" }),
    );
    const body = within(panel).getByLabelText("Procedure (Markdown)");
    await userEvent.clear(body);
    await userEvent.type(body, "## Steps{enter}1. keep this draft");
    // A background refetch (window focus) must not discard the draft.
    window.dispatchEvent(new Event("focus"));
    await userEvent.click(
      within(panel).getByRole("button", { name: "Save edit" }),
    );

    await waitFor(() =>
      expect(
        within(panel).getByLabelText("Procedure (Markdown)"),
      ).toHaveProperty("value", "## Steps\n1. keep this draft"),
    );
    expect(
      (await screen.findAllByText("Blocked by content check")).length,
    ).toBeGreaterThan(0);
  });

  test("submits the base captured when editing began and keeps the draft on a conflict", async () => {
    const v3 = version({
      id: "ver-3",
      version: 3,
      base_version_id: "ver-2",
      origin: "edited",
      origin_label: "Edited by you",
      body: "## Steps\n1. Saved from another tab\n",
    });
    const editBodies: unknown[] = [];
    server.use(
      getEditLearnedSkillMockHandler200(async (info) => {
        editBodies.push(await info.request.json());
        return {
          status: "conflict",
          status_label: "Conflict",
          reason:
            "the skill changed since you started editing; review the current version, then apply your draft again",
          version: null,
          pattern_class: null,
          blocked_step: null,
        };
      }),
    );
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    render(
      <>
        <ExpertDetailPage />
        <Toaster />
      </>,
    );

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    await userEvent.click(
      await within(panel).findByRole("button", { name: "Edit skill" }),
    );
    const body = within(panel).getByLabelText("Procedure (Markdown)");
    await userEvent.clear(body);
    await userEvent.type(body, "## Steps{enter}1. typed over v2");

    // Another tab saves v3 while this editor is still typing; the
    // background refetch must neither discard the draft nor move the base.
    server.use(
      getGetSkillLearningDetailMockHandler200({
        ...detail,
        current_version: v3,
        versions: [v3, v2, v1],
      }),
    );
    window.dispatchEvent(new Event("focus"));
    await userEvent.click(
      within(panel).getByRole("button", { name: "Save edit" }),
    );

    await waitFor(() => expect(editBodies).toHaveLength(1));
    expect(editBodies[0]).toMatchObject({
      expected_version_id: "ver-2",
      body: "## Steps\n1. typed over v2",
    });
    expect((await screen.findAllByText("Conflict")).length).toBeGreaterThan(0);
    expect(within(panel).getByLabelText("Procedure (Markdown)")).toHaveProperty(
      "value",
      "## Steps\n1. typed over v2",
    );
  });

  test("edit is only offered on the current version", async () => {
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    searchParams.set("version", "ver-1");
    render(<ExpertDetailPage />);

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(
      (await within(panel).findByText(/not the current version/)).textContent,
    ).toContain("v1");
    expect(
      within(panel)
        .getByRole("button", { name: "Edit skill" })
        .hasAttribute("disabled"),
    ).toBe(true);
  });

  test("navigating to another version resets the selection to the requested record", async () => {
    server.use(
      getGetSkillLearningDetailMockHandler200({
        ...detail,
        versions: [
          v2,
          {
            ...v1,
            state: "invalidated",
            state_label: "Archived (source unavailable)",
          },
        ],
      }),
    );
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    searchParams.set("version", "ver-1");
    const view = render(<ExpertDetailPage />);

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(
      (await within(panel).findByTestId("version-state")).textContent,
    ).toBe("Archived (source unavailable)");
    expect(
      within(panel)
        .getByRole("button", { name: "Restore v1" })
        .hasAttribute("disabled"),
    ).toBe(true);

    searchParams.set("version", "ver-2");
    view.rerender(<ExpertDetailPage />);
    await waitFor(() =>
      expect(within(panel).getByTestId("version-state").textContent).toBe(
        "Ready to use",
      ),
    );
    expect(
      within(panel).queryByRole("button", { name: /Restore v/ }),
    ).toBeNull();
  });
});

describe("exact version review", () => {
  test("requests an older linked version even when it is outside recent history", async () => {
    const requested: (string | null)[] = [];
    server.use(
      getGetSkillLearningDetailMockHandler200(({ request }) => {
        const id = new URL(request.url).searchParams.get("version_id");
        requested.push(id);
        return { ...detail, versions: id === "ver-1" ? [v2, v1] : [v2] };
      }),
    );
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    searchParams.set("version", "ver-1");
    render(<ExpertDetailPage />);
    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(
      (await within(panel).findByText(/not the current version/)).textContent,
    ).toContain("v1");
    expect(requested).toContain("ver-1");
  });

  test("does not substitute the current version when a requested version is unavailable", async () => {
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    searchParams.set("version", "missing-version");
    render(<ExpertDetailPage />);
    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(await within(panel).findByText(/could not load/i)).toBeDefined();
    expect(
      within(panel).queryByRole("button", { name: "Edit skill" }),
    ).toBeNull();
    expect(within(panel).queryByTestId("version-state")).toBeNull();
  });

  test("shows the step comparison and complete raw changes for reordered procedures", async () => {
    server.use(
      getGetSkillLearningDetailMockHandler200({
        ...detail,
        current_version: { ...v2, body: "## Steps\n1. Import\n2. Validate" },
        versions: [
          { ...v2, body: "## Steps\n1. Import\n2. Validate" },
          { ...v1, body: "## Steps\n1. Validate\n2. Import" },
        ],
      }),
    );
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    render(<ExpertDetailPage />);
    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    await userEvent.click(
      await within(panel).findByRole("tab", { name: "Changes" }),
    );
    expect(within(panel).getByText("Compared with v1")).toBeDefined();
    const changes = within(panel).getByRole("list", {
      name: "Changes by step",
    });
    expect(within(changes).getAllByRole("listitem")).toHaveLength(2);
    await userEvent.click(
      within(panel).getByRole("button", { name: "Show raw Markdown diff" }),
    );
    expect(panel.querySelector("pre")?.textContent).toBe(
      "- 1. Validate\n- 2. Import\n+ 1. Import\n+ 2. Validate",
    );
    await userEvent.click(
      within(panel).getByRole("button", { name: "Hide raw diff" }),
    );
    expect(panel.querySelector("pre")).toBeNull();
  });
});

describe("recovering an unfinished edit", () => {
  test("keeps the procedure, description, and improvement choice after closing the panel", async () => {
    render(<ExpertDetailPage />);
    await openSkillsTab();
    await userEvent.click(
      await screen.findByRole("button", {
        name: "Learning details for csv-import-checks",
      }),
    );
    let panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    await userEvent.click(
      await within(panel).findByRole("button", { name: "Edit skill" }),
    );
    await userEvent.clear(within(panel).getByLabelText("Description"));
    await userEvent.type(
      within(panel).getByLabelText("Description"),
      "Check before importing",
    );
    await userEvent.clear(within(panel).getByLabelText("Procedure (Markdown)"));
    await userEvent.type(
      within(panel).getByLabelText("Procedure (Markdown)"),
      "1. Keep my correction",
    );
    await userEvent.click(
      within(panel).getByRole("switch", {
        name: "Keep automatic improvements on",
      }),
    );
    await userEvent.click(
      within(panel).getByRole("button", { name: "Close skill learning panel" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "csv-import-checks" }),
      ).toBeNull(),
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: "Learning details for csv-import-checks",
      }),
    );
    panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(
      await within(panel).findByLabelText("Procedure (Markdown)"),
    ).toHaveProperty("value", "1. Keep my correction");
    expect(within(panel).getByLabelText("Description")).toHaveProperty(
      "value",
      "Check before importing",
    );
    expect(
      within(panel)
        .getByRole("switch", { name: "Keep automatic improvements on" })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  test("lets the owner review a newer version and explicitly retry the preserved draft", async () => {
    const v3 = version({
      id: "ver-3",
      version: 3,
      base_version_id: "ver-2",
      description: "Check tab-separated input too",
      triggers: ["CSV import", "TSV import"],
      body: "## Steps\n1. Saved from another tab",
    });
    const submissions: unknown[] = [];
    let current = detail;
    server.use(
      getGetSkillLearningDetailMockHandler200(() => current),
      getEditLearnedSkillMockHandler200(async ({ request }) => {
        submissions.push(await request.json());
        current = { ...detail, current_version: v3, versions: [v3, v2, v1] };
        return {
          status: submissions.length === 1 ? "conflict" : "applied",
          status_label: submissions.length === 1 ? "Conflict" : "Applied",
          reason: "A newer version was saved",
          version: null,
          pattern_class: null,
          blocked_step: null,
        };
      }),
    );
    searchParams.set("tab", "skills");
    searchParams.set("skill", "csv-import-checks");
    render(<ExpertDetailPage />);
    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    await userEvent.click(
      await within(panel).findByRole("button", { name: "Edit skill" }),
    );
    await userEvent.clear(within(panel).getByLabelText("Procedure (Markdown)"));
    await userEvent.type(
      within(panel).getByLabelText("Procedure (Markdown)"),
      "1. My correction",
    );
    await userEvent.click(
      within(panel).getByRole("button", { name: "Save edit" }),
    );
    await waitFor(() => expect(submissions).toHaveLength(1));
    expect(submissions[0]).toMatchObject({ expected_version_id: "ver-2" });
    const review = await within(panel).findByText("Review changes against v3");
    expect(
      within(panel)
        .getByRole("button", { name: "Save edit" })
        .hasAttribute("disabled"),
    ).toBe(true);
    await userEvent.click(review);
    expect(
      within(panel).getByText(
        "Current description: Check tab-separated input too",
      ),
    ).toBeDefined();
    expect(panel.querySelector("pre")?.textContent).toContain(
      "Saved from another tab",
    );
    await userEvent.click(
      within(panel).getByRole("button", { name: "Continue editing from v3" }),
    );
    await userEvent.click(
      within(panel).getByRole("button", { name: "Save edit" }),
    );
    await waitFor(() => expect(submissions).toHaveLength(2));
    expect(submissions[1]).toMatchObject({
      expected_version_id: "ver-3",
      body: "1. My correction",
      triggers: ["CSV import", "TSV import"],
    });
    await waitFor(() =>
      expect(within(panel).queryByLabelText("Procedure (Markdown)")).toBeNull(),
    );
  });
});

test("an edited alternative stays with its proposal when a new proposal arrives", async () => {
  const first = version({
    id: "proposal-1",
    version: 3,
    state: "needs_decision",
    state_label: "Needs your decision",
    base_version_id: "ver-2",
  });
  const second = version({
    ...first,
    id: "proposal-2",
    version: 4,
    body: "## Steps\n1. Check delimiters",
  });
  let proposal = first;
  server.use(
    getGetSkillLearningDetailMockHandler200(() => ({
      ...detail,
      open_decision: proposal,
    })),
    getUpdateSkillLearningPolicyMockHandler200(() => detail.policy),
  );
  searchParams.set("tab", "skills");
  searchParams.set("skill", "csv-import-checks");
  render(<ExpertDetailPage />);
  const panel = await screen.findByRole("complementary", {
    name: "csv-import-checks",
  });
  const input = await within(panel).findByLabelText(
    "Edited alternative (optional)",
  );
  await userEvent.type(input, "1. Alternative for proposal one");
  await userEvent.click(
    within(panel).getByRole("button", { name: "Pause learning" }),
  );
  expect(
    within(panel).getByLabelText("Edited alternative (optional)"),
  ).toHaveProperty("value", "1. Alternative for proposal one");
  proposal = second;
  await userEvent.click(
    within(panel).getByRole("button", { name: "Pause learning" }),
  );
  await waitFor(() =>
    expect(
      within(panel).getByLabelText("Edited alternative (optional)"),
    ).toHaveProperty("value", ""),
  );
  expect(
    within(panel)
      .getByRole("button", { name: "Apply edited alternative" })
      .hasAttribute("disabled"),
  ).toBe(true);
  await userEvent.click(
    within(panel).getByRole("button", { name: "Bring back my draft" }),
  );
  expect(
    within(panel).getByLabelText("Edited alternative (optional)"),
  ).toHaveProperty("value", "1. Alternative for proposal one");
  expect(
    within(panel).queryByRole("button", { name: "Bring back my draft" }),
  ).toBeNull();
});
