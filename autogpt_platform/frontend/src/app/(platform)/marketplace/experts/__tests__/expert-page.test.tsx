import { getListExpertTemplatesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import MarketplaceExpertPage from "../[expertId]/page";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockParams = vi.hoisted(() => ({ expertId: "template-maria" }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/experts/template-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => mockParams,
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist with fifteen years across B2B SaaS.",
  skills: ["Content strategy", "Positioning"],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [
    {
      id: "wf-1",
      name: "LinkedIn Post Generator",
      description: "Create research-driven LinkedIn posts in minutes.",
      store_listing_version_id: null,
      library_agent_id: null,
      graph_id: null,
    },
  ],
};

describe("Marketplace expert page", () => {
  beforeEach(() => {
    mockParams.expertId = "template-maria";
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(getListExpertTemplatesMockHandler([mariaTemplate]));
  });

  test("shows the profile with a coming-soon label and no hire action", async () => {
    render(<MarketplaceExpertPage />);

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(screen.getByText("Grows your brand while you sleep")).toBeDefined();
    expect(screen.getByText("Content strategy")).toBeDefined();
    expect(screen.getByText("LinkedIn Post Generator")).toBeDefined();
    expect(screen.getByText("Coming soon")).toBeDefined();
    expect(screen.queryByRole("button", { name: /hire/i })).toBeNull();
    expect(screen.queryByRole("link", { name: "Get started" })).toBeNull();
    expect(
      screen
        .getByRole("link", { name: "Back to marketplace" })
        .getAttribute("href"),
    ).toBe("/marketplace#experts");
  });

  test("shows the same profile and label to signed-out visitors", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });

    render(<MarketplaceExpertPage />);

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(screen.getByText("Coming soon")).toBeDefined();
    expect(screen.queryByRole("link", { name: "Get started" })).toBeNull();
    expect(screen.queryByRole("button", { name: /hire/i })).toBeNull();
  });
});
