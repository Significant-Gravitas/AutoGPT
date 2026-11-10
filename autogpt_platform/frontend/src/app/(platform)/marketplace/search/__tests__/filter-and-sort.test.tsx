import {
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { server } from "@/mocks/mock-server";
import {
  configure,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { MainSearchResultPage } from "../components/MainSearchResultPage/MainSearchResultPage";

configure({ asyncUtilTimeout: 10000 });

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
  usePathname: () => "/marketplace/search",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

const outreach: MarketplaceSkill = {
  slug: "outreach-playbook",
  name: "outreach-playbook",
  title: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: [],
  install_count: 3,
  creator: null,
  creator_avatar: null,
};

describe("Marketplace search filters and sort", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(),
      getGetV2ListStoreCreatorsMockHandler(),
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
  });

  test("a filter chip is pressed once picked and the others are not", async () => {
    render(<MainSearchResultPage searchTerm="outreach" sort="runs" />);

    const all = await screen.findByRole("button", { name: /^All/ });
    const skills = screen.getByRole("button", { name: /^Skills/ });
    expect(all.getAttribute("aria-pressed")).toBe("true");
    expect(skills.getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(skills);

    expect(skills.getAttribute("aria-pressed")).toBe("true");
    expect(all.getAttribute("aria-pressed")).toBe("false");
    expect(
      await screen.findByRole("heading", { name: "Skills" }),
    ).toBeDefined();
  });

  test("the sort menu shows the option picked from it", async () => {
    render(<MainSearchResultPage searchTerm="outreach" sort="runs" />);

    const trigger = await screen.findByRole("button", { name: /Sort by/ });
    expect(trigger.textContent).toContain("Most Runs");

    // Radix DropdownMenu opens on pointerdown, not click, under happy-dom.
    fireEvent.pointerDown(trigger, { button: 0 });
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Name (A-Z)" }),
    );

    expect(
      (await screen.findByRole("button", { name: /Sort by/ })).textContent,
    ).toContain("Name (A-Z)");
  });
});
