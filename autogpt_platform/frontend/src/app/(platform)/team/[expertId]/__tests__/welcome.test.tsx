import {
  getDraftExpertSoulMockHandler,
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getGetRaiseIntroMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetHomeDashboardMockHandler,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import { getListMyExpertMemoryFactsMockHandler200 } from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { RaiseIntro } from "@/app/api/__generated__/models/raiseIntro";
import type { SoulDraftRequest } from "@/app/api/__generated__/models/soulDraftRequest";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

const navigation = vi.hoisted(() => ({ search: "welcome=1" }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team/expert-deps",
  useSearchParams: () => new URLSearchParams(navigation.search),
  useParams: () => ({ expertId: "expert-deps" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

const deps: Expert = {
  id: "expert-deps",
  name: "Deps merger",
  avatar_url: null,
  role: "Developer",
  job_title: "Release engineer",
  tagline: null,
  bio: null,
  skills: [],
  identity: "I'm Deps merger, an AI Expert created by you.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  weekly_budget: 500,
  weekly_spend: 0,
};

const intro: RaiseIntro = {
  intro: "I'll keep your dependency update PRs reviewed and merged.",
  questions: [
    {
      question: "Which repo should I watch?",
      options: ["The main app", "Every repo", "Just one"],
    },
    {
      question: "Can I merge on my own?",
      options: ["Patch updates only", "Ask me first", "Anything green"],
    },
  ],
  is_generated: true,
};

beforeEach(() => {
  navigation.search = "welcome=1";
  server.use(
    getGetExpertMockHandler(deps),
    getGetRaiseIntroMockHandler(intro),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({ timezone: "UTC", days: [] }),
    getListMyExpertMemoryFactsMockHandler200({ expert_id: deps.id, items: [] }),
    getGetHomeDashboardMockHandler(
      getGetHomeDashboardResponseMock200({ attention: [] }),
    ),
  );
});

afterEach(() => {
  window.localStorage.clear();
});

describe("new expert welcome", () => {
  test("states its job, asks with quick picks and writes the answers into its soul", async () => {
    let drafted: SoulDraftRequest | null = null;
    server.use(
      getDraftExpertSoulMockHandler(async ({ request }) => {
        drafted = (await request.json()) as SoulDraftRequest;
        return { expected_identity: deps.identity };
      }),
    );
    render(<ExpertDetailPage />);

    const panel = await screen.findByRole("complementary", {
      name: "Deps merger introduces themself",
    });
    expect(
      await within(panel).findByText(
        "I'll keep your dependency update PRs reviewed and merged.",
      ),
    ).toBeDefined();

    await userEvent.click(
      within(panel).getByRole("radio", { name: "Every repo" }),
    );
    await userEvent.click(
      within(
        within(panel).getByRole("radiogroup", {
          name: "Can I merge on my own?",
        }),
      ).getByRole("button", { name: "Something else" }),
    );
    await userEvent.type(
      within(panel).getByRole("textbox"),
      "Minor and patch{Enter}",
    );

    expect(
      await within(panel).findByText(/Got it\. I'm writing that into my soul/),
    ).toBeDefined();
    await waitFor(() =>
      expect(drafted).toEqual({
        intro: intro.intro,
        answers: [
          { question: "Which repo should I watch?", answer: "Every repo" },
          { question: "Can I merge on my own?", answer: "Minor and patch" },
        ],
      }),
    );
    expect(within(panel).getByText(/\$5 weekly budget/)).toBeDefined();
  });

  test("skipping asks nothing more and saves nothing", async () => {
    const draft = vi.fn();
    server.use(
      getDraftExpertSoulMockHandler(() => {
        draft();
        return { expected_identity: deps.identity };
      }),
    );
    render(<ExpertDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Skip for now" }),
    );

    expect(
      await screen.findByText(/No problem\. Tell me what you need/),
    ).toBeDefined();
    expect(screen.getAllByRole("radio")[0]).toHaveProperty("disabled", true);
    expect(draft).not.toHaveBeenCalled();
  });

  test("opens settings with the fields the create dialog no longer asks", async () => {
    render(<ExpertDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Skip for now" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Open settings" }),
    );

    const setup = await screen.findByRole("region", {
      name: "Deps merger setup",
    });
    expect(within(setup).getByText("$5 / week")).toBeDefined();
    expect(within(setup).getByText("Voice")).toBeDefined();
    expect(within(setup).getByText("No workflows yet")).toBeDefined();
    expect(within(setup).getByText("No skills yet")).toBeDefined();
    expect(
      screen.queryByRole("complementary", {
        name: "Deps merger introduces themself",
      }),
    ).toBeNull();
  });

  test("goes straight to the hand-off when the opening cannot load", async () => {
    server.use(
      http.get("/api/proxy/api/experts/raise/intro", () =>
        HttpResponse.json({ detail: "Offline" }, { status: 500 }),
      ),
    );
    render(<ExpertDetailPage />);

    expect(
      await screen.findByText(/No problem\. Tell me what you need/, undefined, {
        timeout: 10_000,
      }),
    ).toBeDefined();
    expect(screen.queryByRole("status", { name: "Typing" })).toBeNull();
  });

  test("stays closed without the welcome link", async () => {
    navigation.search = "";
    render(<ExpertDetailPage />);

    expect(await screen.findByText("Deps merger")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
  });
});
