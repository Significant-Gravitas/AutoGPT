import {
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV1CheckIfOnboardingIsCompletedMockHandler200,
  getGetV1OnboardingStateMockHandler200,
  getGetV1OnboardingStateResponseMock200,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

const { capture } = vi.hoisted(() => ({ capture: vi.fn() }));
vi.mock("posthog-js", () => ({ default: { capture } }));

/** Funnel events as PostHog received them, in order. */
function funnelCalls() {
  return capture.mock.calls.map(([event, data]) => ({
    type: event as string,
    data: (data ?? {}) as Record<string, unknown>,
  }));
}

beforeEach(() => {
  capture.mockReset();
});

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

const mariaTemplate: Expert = {
  id: "template-maria",
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
    "The expert asks for approval before acting externally.",
  ],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

const hiredMaria: Expert = {
  ...mariaTemplate,
  id: "expert-maria",
  is_template: false,
  source_template_id: "template-maria",
};

describe("Marketplace ExpertsSection", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
    });
    server.use(
      getGetV1CheckIfOnboardingIsCompletedMockHandler200({
        is_completed: true,
      }),
      getGetV1OnboardingStateMockHandler200(
        getGetV1OnboardingStateResponseMock200({
          userId: "user-1",
          completedSteps: ["ONBOARDING_COMPLETE", "MARKETPLACE_TAB_INTRO"],
        }),
      ),
    );
  });

  test("links each expert card to its own page", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Create an Expert" })
        .getAttribute("href"),
    ).toBe("/raise");
    // The card is the link: a shared URL lands on the same profile the
    // marketplace opens, with no dialog in between.
    const card = await screen.findByRole("link", { name: /Maria/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/experts/template-maria",
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  test("shows the expert cards to signed-out visitors without account links", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });

    let rosterRequested = false;
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler(() => {
        rosterRequested = true;
        return [];
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    const card = await screen.findByRole("link", { name: /Maria/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/experts/template-maria",
    );
    expect(screen.queryByText("Create an Expert")).toBeNull();
    expect(screen.queryByRole("link", { name: "View your team" })).toBeNull();
    expect(
      screen.getByRole("link", { name: "Hire" }).getAttribute("href"),
    ).toBe("/signup?next=%2Fmarketplace%2Fexperts%2Ftemplate-maria");
    expect(rosterRequested).toBe(false);
  });

  test("shows experts to signed-in users without a rollout flag", async () => {
    let templatesRequested = false;
    let expertsRequested = false;
    server.use(
      getListExpertTemplatesMockHandler(() => {
        templatesRequested = true;
        return [mariaTemplate];
      }),
      getListExpertsMockHandler(() => {
        expertsRequested = true;
        return [];
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Create an Expert" }),
    ).toBeDefined();
    expect(await screen.findByRole("link", { name: /Maria/ })).toBeDefined();
    expect(templatesRequested).toBe(true);
    await waitFor(() => expect(expertsRequested).toBe(true));
  });

  test("keeps the raise-your-own door open when no templates exist", async () => {
    server.use(
      getListExpertTemplatesMockHandler([]),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    await waitFor(
      () => {
        const raiseLink = screen.getByRole("link", {
          name: "Create an Expert",
        });
        expect(raiseLink.getAttribute("href")).toBe("/raise");
        expect(screen.queryByText("Meet the AI Experts")).toBeNull();
      },
      { timeout: 5_000 },
    );
  });

  test("keeps the raise button when templates fail to load", async () => {
    server.use(
      http.get("/api/proxy/api/experts/templates", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 500 }),
      ),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    const raiseLink = await screen.findByRole("link", {
      name: "Create an Expert",
    });
    expect(raiseLink.getAttribute("href")).toBe("/raise");
  });

  test("lists the Hub skills a hire comes with on the card, and nothing else", async () => {
    server.use(
      getListExpertTemplatesMockHandler([
        {
          ...mariaTemplate,
          skills: ["Content strategy", "brand-voice-guide"],
          bundled_skills: [
            {
              id: "listing-1",
              slug: "brand-voice-guide",
              title: "Brand voice guide",
              description: "Keeps every draft on-brand.",
            },
          ],
        },
      ]),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    const card = await screen.findByRole(
      "link",
      { name: /Maria/ },
      { timeout: 5_000 },
    );
    expect(within(card).getByText("Brand voice guide")).toBeDefined();
    expect(within(card).queryByText("brand-voice-guide")).toBeNull();
    expect(within(card).queryByText("Content strategy")).toBeNull();
  });

  test("shows the job title on the name line, and the role when there is none", async () => {
    server.use(
      getListExpertTemplatesMockHandler([
        { ...mariaTemplate, job_title: "SEO Content Manager" },
        { ...mariaTemplate, id: "template-max", name: "Max", role: "Sales" },
      ]),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    const maria = await screen.findByRole(
      "link",
      { name: /Maria/ },
      { timeout: 5_000 },
    );
    expect(within(maria).getByText(/SEO Content Manager/)).toBeDefined();
    expect(within(maria).queryByText("Marketing Strategist")).toBeNull();
    expect(within(maria).queryByText("AI Expert")).toBeNull();
    const max = screen.getByRole("link", { name: /Max/ });
    expect(within(max).getByText(/Sales Development Rep/)).toBeDefined();
  });

  test("shows no skills row on a card without Hub skills", async () => {
    server.use(
      getListExpertTemplatesMockHandler([
        { ...mariaTemplate, skills: ["Content strategy"], bundled_skills: [] },
      ]),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    const card = await screen.findByRole(
      "link",
      { name: /Maria/ },
      { timeout: 5_000 },
    );
    expect(within(card).queryByText("Skills")).toBeNull();
    expect(within(card).queryByText("Content strategy")).toBeNull();
  });

  test("hired template shows hired state", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(await screen.findByText("On your team")).toBeDefined();
  });

  test("emits the section view event once the shelf has rendered", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    await screen.findByText("Maria");
    await waitFor(() =>
      expect(funnelCalls().map((body) => body.type)).toContain(
        "experts_section_viewed",
      ),
    );
  });

  test("template becomes viewable again once the expert is fired", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([{ ...hiredMaria, is_archived: true }]),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    await screen.findByText("Maria");
    expect(screen.getByText("View")).toBeDefined();
    expect(screen.queryByText("On your team")).toBeNull();
  });
});
