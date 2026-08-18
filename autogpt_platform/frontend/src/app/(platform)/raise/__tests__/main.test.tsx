import { getCreateRaisedExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RaisePage from "../page";
import { loadDraft, saveDraft, VOICE_SKIPPED_LABEL } from "../helpers";

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
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? setFlagStatusMock()
        : actual.useFlagStatus(flag as never),
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
  identity: "I'm Otto, raised by you. I learn how you work and grow with you.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
} as Expert;

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
  server.use(getListCopilotSkillsMockHandler([]));
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
    weekly_budget: null,
    attachments: [],
  });
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=raised-1&kickoff=1",
    ),
  );
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
  expect(await screen.findByText("Couldn't raise Otto")).toBeDefined();
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

test("distinguishes the lifetime raised-expert limit", async () => {
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

test("picking a weekly budget advances to marketplace workflows", async () => {
  seedAtBudget();
  renderRaise();
  await userEvent.click(
    await screen.findByRole("button", { name: /500 credits/ }),
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
