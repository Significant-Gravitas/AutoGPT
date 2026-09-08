import {
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { getListExpertTemplatesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

const mockUseAuth = vi.hoisted(() => vi.fn());
const flags = vi.hoisted(() => ({ skillsHub: true, hireExperts: false }));

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

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => {
      if (flag === "skills-hub") return flags.skillsHub;
      if (flag === "hire-experts") return flags.hireExperts;
      return actual.useGetFlag(flag as never);
    },
  };
});

const brandVoice: MarketplaceSkill = {
  slug: "brand-voice-guide",
  name: "Brand voice guide",
  description: "Write in a consistent brand voice.",
  categories: ["content"],
  required_providers: [],
  install_count: 12,
  creator: null,
  creator_avatar: null,
};

describe("Marketplace SkillsSection", () => {
  beforeEach(() => {
    flags.skillsHub = true;
    flags.hireExperts = false;
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(),
      getGetV2ListStoreCreatorsMockHandler(),
      getListExpertTemplatesMockHandler([]),
    );
  });

  test("shows skills as their own shelf, linked to the skill page", async () => {
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200({
        skills: [brandVoice],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 6,
        },
      }),
    );

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Skills to teach", undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
    const card = await screen.findByRole("link", { name: /Brand voice guide/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/skills/brand-voice-guide",
    );
  });

  test("shows the compatibility line only for a skill that needs one", async () => {
    const withProvider: MarketplaceSkill = {
      ...brandVoice,
      slug: "outreach-playbook",
      name: "Outreach playbook",
      required_providers: ["google"],
    };
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200({
        skills: [brandVoice, withProvider],
        pagination: {
          total_items: 2,
          total_pages: 1,
          current_page: 1,
          page_size: 6,
        },
      }),
    );

    render(<MainMarkeplacePage />);

    await screen.findAllByTestId("skill-card", undefined, { timeout: 10000 });
    const outreach = await screen.findByRole("link", {
      name: /Outreach playbook/,
    });
    expect(outreach.textContent).toContain("Works with Google");
    const brand = await screen.findByRole("link", {
      name: /Brand voice guide/,
    });
    expect(brand.textContent).not.toContain("Works with");
  });

  test("says nothing at all when the marketplace has no skills yet", async () => {
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200({
        skills: [],
        pagination: {
          total_items: 0,
          total_pages: 0,
          current_page: 1,
          page_size: 6,
        },
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("Skills to teach")).toBeNull(),
    );
  });

  test("stays hidden and fetches nothing outside the beta", async () => {
    flags.skillsHub = false;
    let requested = false;
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200(() => {
        requested = true;
        return {
          skills: [brandVoice],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 6,
          },
        };
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    expect(screen.queryByText("Skills to teach")).toBeNull();
    await waitFor(() => expect(requested).toBe(false));
  });
});
