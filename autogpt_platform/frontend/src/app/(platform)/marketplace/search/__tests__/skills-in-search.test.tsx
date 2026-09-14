import {
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { MainSearchResultPage } from "../components/MainSearchResultPage/MainSearchResultPage";

// These pages wait on several queries before anything renders, and CI is
// slower than a dev machine; the testing-library default is one second.
configure({ asyncUtilTimeout: 10000 });

const mockUseAuth = vi.hoisted(() => vi.fn());
const flag = vi.hoisted(() => ({ enabled: true, ready: true }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/search",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useFlagStatus: () => flag };
});

const outreach: MarketplaceSkill = {
  slug: "outreach-playbook",
  name: "outreach-playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: [],
  install_count: 3,
  creator: null,
  creator_avatar: null,
};

describe("Skills in marketplace search", () => {
  beforeEach(() => {
    flag.enabled = true;
    flag.ready = true;
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(),
      getGetV2ListStoreCreatorsMockHandler(),
    );
  });

  test("returns matching skills alongside agents and creators", async () => {
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200({
        skills: [outreach],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    );

    render(<MainSearchResultPage searchTerm="outreach" sort="runs" />);

    expect(
      await screen.findByRole(
        "heading",
        { name: "Skills" },
        { timeout: 10000 },
      ),
    ).toBeDefined();
    const card = await screen.findByRole("link", { name: /Outreach playbook/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/skills/outreach-playbook",
    );
    expect(screen.getByRole("button", { name: /Skills/ })).toBeDefined();
  });

  test("asks for no skills and offers no chip outside the beta", async () => {
    flag.enabled = false;
    let requested = false;
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200(() => {
        requested = true;
        return {
          skills: [outreach],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 20,
          },
        };
      }),
    );

    render(<MainSearchResultPage searchTerm="outreach" sort="runs" />);

    await screen.findByText(/Showing results for/, undefined, {
      timeout: 10000,
    });
    await waitFor(() => expect(requested).toBe(false));
    expect(screen.queryByRole("heading", { name: "Skills" })).toBeNull();
  });
});
