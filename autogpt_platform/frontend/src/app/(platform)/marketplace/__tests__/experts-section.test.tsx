import {
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

const mockUseAuth = vi.hoisted(() => vi.fn());
const hireExpertsFlag = vi.hoisted(() => ({ enabled: true }));

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

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "hire-experts"
        ? hireExpertsFlag.enabled
        : actual.useGetFlag(flag as never),
  };
});

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
    "External actions require approval.",
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
    hireExpertsFlag.enabled = true;
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
    });
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
        .getByRole("link", { name: /raise your own expert from scratch/i })
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
    hireExpertsFlag.enabled = false;
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
    expect(screen.queryByText(/raise your own expert/i)).toBeNull();
    expect(screen.queryByRole("link", { name: "View your team" })).toBeNull();
    expect(rosterRequested).toBe(false);
  });

  test("stays hidden and fetches nothing for signed-in users outside the beta", async () => {
    hireExpertsFlag.enabled = false;
    let templatesRequested = false;
    server.use(
      getListExpertTemplatesMockHandler(() => {
        templatesRequested = true;
        return [mariaTemplate];
      }),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    expect(screen.queryByText("Meet the AI Experts")).toBeNull();
    expect(screen.queryByText(/raise your own expert/)).toBeNull();
    expect(templatesRequested).toBe(false);
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
          name: "Raise your own expert from scratch",
        });
        expect(raiseLink.getAttribute("href")).toBe("/raise");
        expect(raiseLink.textContent).not.toContain("…or");
        expect(screen.queryByText("Meet the AI Experts")).toBeNull();
      },
      { timeout: 5_000 },
    );
  });

  test("uses standalone raise copy when templates fail to load", async () => {
    server.use(
      http.get("/api/proxy/api/experts/templates", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 500 }),
      ),
      getListExpertsMockHandler([]),
    );

    render(<MainMarkeplacePage />);

    const raiseLink = await screen.findByRole("link", {
      name: "Raise your own expert from scratch",
    });
    expect(raiseLink.getAttribute("href")).toBe("/raise");
    expect(raiseLink.textContent).not.toContain("…or");
  });

  test("hired template shows hired state", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(await screen.findByText("Hired")).toBeDefined();
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
    expect(screen.queryByText("Hired")).toBeNull();
  });
});
