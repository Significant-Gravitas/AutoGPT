import { getCreateRaisedExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { getGetV2ListMarketplaceSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RaisePage from "../page";
import {
  EMPTY_DRAFT,
  loadDraft,
  saveDraft,
  VOICE_SKIPPED_LABEL,
} from "../helpers";

const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) => {
      if (flag === "hire-experts") return setFlagStatusMock();
      // The Hub is on here so the flow runs in the configuration production
      // will be in: leaving it to the real hook disabled the listing query
      // and left "Hub on and empty" — the case that drops the beat — untested.
      if (flag === "skills-hub") return { enabled: true, ready: true };
      return actual.useFlagStatus(flag as never);
    },
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, user: { id: "user-1" } }),
}));

const { pushMock, notFoundMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  notFoundMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
  usePathname: () => "/raise",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const raisedExpert = {
  id: "raised-1",
  name: "Otto",
  avatar_url: null,
  role: "marketer",
  tagline: null,
  bio: null,
  skills: [],
  identity:
    "I'm Otto, an AI Expert created by you. I use your instructions to help with your work.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
} as Expert;

const LIBRARY_SKILL = { name: "seo-audit", description: "Audit landing pages" };

function hubSkills(skills: MarketplaceSkill[]) {
  return getGetV2ListMarketplaceSkillsMockHandler200({
    skills,
    pagination: {
      total_items: skills.length,
      total_pages: 1,
      current_page: 1,
      page_size: 3,
    },
  });
}

function raiseResult(overrides: Partial<RaiseResult> = {}): RaiseResult {
  return { expert: raisedExpert, failed_attachments: [], ...overrides };
}

function renderRaise() {
  return render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
}

function mockReducedMotion() {
  vi.spyOn(window, "matchMedia").mockImplementation((query) => {
    return {
      matches: query.includes("prefers-reduced-motion"),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    } as unknown as MediaQueryList;
  });
}

function seedAtBudget(name = "Otto") {
  saveDraft({
    step: "budget",
    hasStarted: true,
    role: "marketer",
    jobTitle: "Marketing Manager",
    name,
    color: "rose-300",
    avatarUrl: "",
    about: "",
    voicePreferences: "",
    voiceLabel: VOICE_SKIPPED_LABEL,
    budget: null,
    marketplace: null,
    skills: null,
  });
}

function seedAtSkills(
  name = "Otto",
  budget: { credits: number | null } = { credits: null },
) {
  saveDraft({
    step: "skills",
    hasStarted: true,
    role: "marketer",
    jobTitle: "Marketing Manager",
    name,
    color: "rose-300",
    avatarUrl: "",
    about: "",
    voicePreferences: "",
    voiceLabel: VOICE_SKIPPED_LABEL,
    budget,
    marketplace: [],
    skills: null,
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
  mockReducedMotion();
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  pushMock.mockClear();
  notFoundMock.mockClear();
  // One library skill is enough to keep the skills beat in the flow; with
  // none and an empty Hub, the marketplace beat becomes the last one.
  server.use(getListCopilotSkillsMockHandler([LIBRARY_SKILL]), hubSkills([]));
});

afterEach(() => {
  vi.clearAllMocks();
});

test("calls notFound when the experts feature is disabled", () => {
  setFlagStatusMock.mockReturnValue({ enabled: false, ready: true });

  try {
    renderRaise();
  } catch {}

  expect(notFoundMock).toHaveBeenCalled();
});

test("skips remaining kit steps, posts null budget and empty attachments, and opens copilot", async () => {
  let captured: unknown = null;
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raiseResult();
    }),
  );

  seedAtSkills();
  renderRaise();
  // The skills step only renders its actions once the copilot-skills request
  // settles, which can outrun the 1s default when the suite runs under load.
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip" }, { timeout: 5000 }),
  );

  await waitFor(() => expect(captured).not.toBeNull());
  expect(captured).toMatchObject({
    name: "Otto",
    role: "marketer",
    job_title: "Marketing Manager",
    weekly_budget: null,
    attachments: [],
  });
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=raised-1&kickoff=1",
    ),
  );
});

test("posts null when the job title was skipped", async () => {
  let captured: unknown = null;
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raiseResult();
    }),
  );

  seedAtSkills();
  saveDraft({ ...loadDraft(), jobTitle: "" });
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: /Bring Otto to life/ }),
  );

  await waitFor(() => expect(captured).toMatchObject({ job_title: null }));
});

test("posts a chosen weekly budget", async () => {
  let captured: unknown = null;
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raiseResult();
    }),
  );

  seedAtSkills("Otto", { credits: 500 });
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: /Bring Otto to life/ }),
  );

  await waitFor(() => expect(captured).not.toBeNull());
  expect(captured).toMatchObject({
    name: "Otto",
    weekly_budget: 500,
    attachments: [],
  });
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=raised-1&kickoff=1",
    ),
  );
});

test("a rapid double-click on finish sends a single POST", async () => {
  let postCount = 0;
  server.use(
    getCreateRaisedExpertMockHandler(() => {
      postCount += 1;
      return raiseResult();
    }),
  );

  seedAtSkills();
  renderRaise();
  const finishButton = await screen.findByRole("button", { name: /life/ });
  await Promise.all([
    userEvent.click(finishButton),
    userEvent.click(finishButton),
  ]);

  await waitFor(() => expect(pushMock).toHaveBeenCalled());
  expect(postCount).toBe(1);
});

test("unlocks finish after a raise POST fails so the user can retry", async () => {
  let postCount = 0;
  server.use(
    http.post("/api/proxy/api/experts/raise", () => {
      postCount += 1;
      return HttpResponse.json({ detail: "Raise failed" }, { status: 500 });
    }),
  );

  seedAtSkills();
  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));
  expect(await screen.findByText("Couldn't create Otto")).toBeDefined();
  await waitFor(() => expect(postCount).toBe(1));

  const retry = screen.getByRole("button", { name: /Bring Otto to life/ });
  expect((retry as HTMLButtonElement).disabled).toBe(false);
  await userEvent.click(retry);
  await waitFor(() => expect(postCount).toBe(2));
});

test("shows a friendly limit message on 409", async () => {
  server.use(
    http.post("/api/proxy/api/experts/raise", () =>
      HttpResponse.json(
        { detail: { code: "active_expert_limit", limit: 20 } },
        { status: 409 },
      ),
    ),
  );

  seedAtSkills();
  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  expect(await screen.findByText("Your team is full")).toBeDefined();
  expect(pushMock).not.toHaveBeenCalled();
});

test("distinguishes the lifetime limit for created Experts", async () => {
  server.use(
    http.post("/api/proxy/api/experts/raise", () =>
      HttpResponse.json(
        { detail: { code: "raised_expert_lifetime_limit", limit: 100 } },
        { status: 409 },
      ),
    ),
  );

  seedAtSkills();
  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  expect(
    await screen.findByText("Expert creation limit reached"),
  ).toBeDefined();
  expect(screen.getByText(/Contact support/)).toBeDefined();
  expect(pushMock).not.toHaveBeenCalled();
});

test("toasts failed attachments and still opens copilot", async () => {
  server.use(
    getCreateRaisedExpertMockHandler(
      raiseResult({
        failed_attachments: [
          {
            kind: "workflow",
            source: "marketplace",
            id: "listing-1",
            reason: "installation_failed",
          },
        ],
      }),
    ),
  );

  seedAtSkills();
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: /Bring Otto to life/ }),
  );

  expect(await screen.findByText(/some tools didn't attach/)).toBeDefined();
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=raised-1&kickoff=1",
    ),
  );
});

test("picking a job title records it and asks for a name", async () => {
  saveDraft({
    ...EMPTY_DRAFT,
    hasStarted: true,
    role: "marketer",
    step: "jobTitle",
  });
  renderRaise();
  await userEvent.click(
    await screen.findByRole(
      "button",
      { name: "Marketing Manager" },
      { timeout: 5000 },
    ),
  );

  expect(
    await screen.findByRole(
      "group",
      { name: "Suggested names" },
      { timeout: 5000 },
    ),
  ).toBeDefined();
  expect(loadDraft()).toMatchObject({
    jobTitle: "Marketing Manager",
    step: "name",
  });
});

test("typing a job title trims it and asks for a name", async () => {
  saveDraft({
    ...EMPTY_DRAFT,
    hasStarted: true,
    role: "Custom role",
    step: "jobTitle",
  });
  renderRaise();
  await userEvent.type(
    await screen.findByRole("textbox", { name: "Job title" }),
    "  Chief of Staff  ",
  );
  await userEvent.click(screen.getByRole("button", { name: "Add title" }));

  expect(
    await screen.findByRole(
      "group",
      { name: "Suggested names" },
      { timeout: 5000 },
    ),
  ).toBeDefined();
  expect(loadDraft()).toMatchObject({
    jobTitle: "Chief of Staff",
    step: "name",
  });
});

test("skipping a job title records it and asks for a name", async () => {
  saveDraft({
    ...EMPTY_DRAFT,
    hasStarted: true,
    role: "marketer",
    step: "jobTitle",
  });
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip" }, { timeout: 5000 }),
  );

  expect(await screen.findByText("Skipped")).toBeDefined();
  expect(
    await screen.findByRole(
      "group",
      { name: "Suggested names" },
      { timeout: 5000 },
    ),
  ).toBeDefined();
  expect(loadDraft()).toMatchObject({ jobTitle: "", step: "name" });
});

test("picking a weekly budget advances to marketplace workflows", async () => {
  seedAtBudget();
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: "$5 / week" }),
  );

  expect(
    await screen.findByRole(
      "textbox",
      { name: "Search marketplace and library workflows" },
      { timeout: 3000 },
    ),
  ).toBeDefined();
  expect(screen.getByRole("button", { name: "That's it" })).toBeDefined();
});

test("drops the skills beat and raises from the marketplace step when there is nothing to add", async () => {
  let captured: unknown = null;
  // The Hub is on for this file, so this is an empty catalogue plus an empty
  // library — not a disabled flag.
  server.use(
    getListCopilotSkillsMockHandler([]),
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raiseResult();
    }),
  );

  seedAtBudget();
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: "$5 / week" }),
  );

  const finish = await screen.findByRole(
    "button",
    { name: /Bring Otto to life/ },
    { timeout: 5000 },
  );
  expect(screen.queryByRole("textbox", { name: "Search skills" })).toBeNull();
  await userEvent.click(finish);

  await waitFor(() => expect(captured).not.toBeNull());
  expect(captured).toMatchObject({ weekly_budget: 500, attachments: [] });
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=raised-1&kickoff=1",
    ),
  );
});

test("keeps the skills beat when only the Hub has something to offer", async () => {
  server.use(
    getListCopilotSkillsMockHandler([]),
    hubSkills([
      {
        slug: "cold-outreach",
        name: "cold-outreach",
        title: "Cold Outreach",
        description: "Write cold emails",
        categories: ["sales"],
        required_providers: [],
        install_count: 2,
      },
    ]),
  );

  seedAtBudget();
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: "$5 / week" }),
  );

  expect(
    await screen.findByRole("button", { name: "That's it" }, { timeout: 5000 }),
  ).toBeDefined();
  expect(
    screen.queryByRole("button", { name: /Bring Otto to life/ }),
  ).toBeNull();
});

test("keeps the skills beat when availability settles empty after marketplace submit", async () => {
  let captured: unknown = null;
  let settleLibrary!: (skills: (typeof LIBRARY_SKILL)[]) => void;
  const pendingLibrary = new Promise<(typeof LIBRARY_SKILL)[]>((resolve) => {
    settleLibrary = resolve;
  });
  server.use(
    getListCopilotSkillsMockHandler(() => pendingLibrary),
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raiseResult();
    }),
  );

  seedAtBudget();
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: "$5 / week" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "That's it" }),
  );
  settleLibrary([]);

  await userEvent.click(
    await screen.findByRole(
      "button",
      { name: /Bring Otto to life/ },
      { timeout: 5000 },
    ),
  );

  await waitFor(() => expect(captured).not.toBeNull());
  expect(captured).toMatchObject({ weekly_budget: 500, attachments: [] });
});

test("back returns to the previous step and the draft survives", async () => {
  seedAtBudget();
  renderRaise();
  expect(await screen.findByRole("button", { name: "Skip" })).toBeDefined();

  await userEvent.click(screen.getByRole("button", { name: "Back" }));
  expect(
    await screen.findByRole("button", { name: "Skip for now" }),
  ).toBeDefined();

  const draft = loadDraft();
  expect(draft.step).toBe("voice");
  expect(draft.voiceLabel).toBeNull();
  expect(draft).toMatchObject({
    hasStarted: true,
    role: "marketer",
    jobTitle: "Marketing Manager",
    name: "Otto",
    color: "rose-300",
  });
});

test("a refresh resumes the draft from session storage", async () => {
  seedAtSkills("Nova");
  const first = renderRaise();
  expect(
    await screen.findByRole("button", { name: /Bring Nova to life/ }),
  ).toBeDefined();

  first.unmount();
  renderRaise();

  expect(
    await screen.findByRole("button", { name: /Bring Nova to life/ }),
  ).toBeDefined();
});

test.each([400, 422, 503])(
  "keeps the draft and displays the recovery message when creation fails (%s)",
  async (status) => {
    server.use(
      http.post("*/api/experts/raise", () =>
        HttpResponse.json(
          {
            detail:
              "Upload this image again through the appearance picker so it can be reviewed.",
          },
          { status },
        ),
      ),
    );
    seedAtSkills();
    const draft = loadDraft();
    renderRaise();
    const finish = await screen.findByRole("button", {
      name: /Bring Otto to life/,
    });
    await userEvent.click(finish);
    expect(
      await screen.findByText(
        "Upload this image again through the appearance picker so it can be reviewed.",
      ),
    ).toBeDefined();
    expect(loadDraft()).toEqual(draft);
    expect(pushMock).not.toHaveBeenCalled();
    expect(finish.hasAttribute("disabled")).toBe(false);
  },
);
