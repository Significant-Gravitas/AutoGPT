import { getListExpertTemplatesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2ListStoreCategoriesMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

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

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

const CATEGORIES = [
  { value: "marketing", label: "Marketing", description: "Campaigns and SEO" },
  { value: "sales", label: "Sales", description: "Leads and outreach" },
];

function agent(name: string) {
  return {
    slug: name.toLowerCase(),
    agent_name: name,
    agent_image: "",
    creator: "creator",
    creator_avatar: "",
    sub_heading: "",
    description: "",
    runs: 1,
    rating: 5,
    agent_graph_id: `graph-${name}`,
  };
}

/** Records every ?category= the page asks for, and answers from BY_CATEGORY. */
const requestedCategories: (string | null)[] = [];
const BY_CATEGORY: Record<string, string[]> = {
  all: ["Blog Writer", "Lead Finder"],
  marketing: ["Blog Writer"],
};

describe("Marketplace category filter", () => {
  beforeEach(() => {
    requestedCategories.length = 0;
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getListExpertTemplatesMockHandler([]),
      getGetV2ListStoreCreatorsMockHandler({
        creators: [],
        pagination: {
          current_page: 1,
          total_items: 0,
          total_pages: 0,
          page_size: 20,
        },
      }),
      getGetV2ListStoreCategoriesMockHandler(CATEGORIES),
      http.get("/api/proxy/api/store/agents", ({ request }) => {
        const params = new URL(request.url).searchParams;
        // The featured shelf is a separate query; keep it empty so the grid
        // is the only thing rendering agent names.
        const names = params.get("featured")
          ? []
          : (BY_CATEGORY[params.get("category") ?? "all"] ?? []);
        if (!params.get("featured")) {
          requestedCategories.push(params.get("category"));
        }
        return HttpResponse.json({
          agents: names.map(agent),
          pagination: {
            current_page: 1,
            total_items: names.length,
            total_pages: 1,
            page_size: 20,
          },
        });
      }),
    );
  });

  test("filtering by a category asks the API for it and narrows the grid", async () => {
    render(<MainMarkeplacePage />);

    expect((await screen.findAllByText("Lead Finder")).length).toBeGreaterThan(
      0,
    );
    expect(requestedCategories).toEqual([null]);

    await userEvent.click(await findCategoryChip("Marketing"));

    await waitFor(() =>
      expect(screen.queryAllByText("Lead Finder")).toHaveLength(0),
    );
    expect(screen.queryAllByText("Blog Writer").length).toBeGreaterThan(0);
    expect(requestedCategories).toContain("marketing");
  });

  test("clicking the active category again clears the filter", async () => {
    render(<MainMarkeplacePage />);
    expect((await screen.findAllByText("Lead Finder")).length).toBeGreaterThan(
      0,
    );

    const marketing = await findCategoryChip("Marketing");
    await userEvent.click(marketing);
    await waitFor(() =>
      expect(screen.queryAllByText("Lead Finder")).toHaveLength(0),
    );
    expect(marketing.getAttribute("aria-pressed")).toBe("true");

    await userEvent.click(marketing);

    await waitFor(() =>
      expect(screen.queryAllByText("Lead Finder").length).toBeGreaterThan(0),
    );
    expect(marketing.getAttribute("aria-pressed")).toBe("false");
  });

  test("only the eight canonical categories the API serves are offered", async () => {
    render(<MainMarkeplacePage />);

    expect(await findCategoryChip("Marketing")).toBeDefined();
    expect(await findCategoryChip("Sales")).toBeDefined();
    // The legacy free-text options are gone.
    const chips = within(
      await screen.findByRole("group", { name: "Browse by category" }),
    );
    expect(chips.queryByRole("button", { name: "Productivity" })).toBeNull();
    expect(chips.queryByRole("button", { name: /Writing/ })).toBeNull();
  });
});

/** The hero has its own chip row with overlapping labels, so scope to ours. */
async function findCategoryChip(name: string) {
  const group = await screen.findByRole("group", {
    name: "Browse by category",
  });
  return within(group).findByRole("button", { name });
}
