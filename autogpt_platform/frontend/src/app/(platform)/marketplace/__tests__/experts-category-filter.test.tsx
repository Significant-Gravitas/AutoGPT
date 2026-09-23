import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListStoreCategoriesMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

configure({ asyncUtilTimeout: 10000 });

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

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

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

const CATEGORIES = [
  { value: "marketing", label: "Marketing", description: "Campaigns and SEO" },
  { value: "sales", label: "Sales", description: "Leads and outreach" },
];

function template(name: string, categories: string[]): Expert {
  return {
    id: `template-${name.toLowerCase()}`,
    name,
    avatar_url: null,
    role: categories[0],
    bio: null,
    skills: [],
    categories,
    tagline: `${name} works here`,
    identity: `You are ${name}.`,
    voice_preferences: "Direct.",
    boundaries: "Ask first.",
    protected_soul_rules: [],
    is_template: true,
    source_template_id: null,
    is_archived: false,
    workflows: [],
  };
}

const ROSTER = [template("Maria", ["marketing"]), template("Max", ["sales"])];

/** Records every ?category= the roster is asked for, and answers with the
 *  experts filed under it — the server-side filter, as the page sees it. */
const requestedCategories: (string | null)[] = [];

describe("Marketplace category filter over experts", () => {
  beforeEach(() => {
    requestedCategories.length = 0;
    hireExpertsFlag.enabled = true;
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getListExpertsMockHandler([]),
      getGetV2ListStoreCategoriesMockHandler(CATEGORIES),
      http.get("/api/proxy/api/experts/templates", ({ request }) => {
        const category = new URL(request.url).searchParams.get("category");
        requestedCategories.push(category);
        return HttpResponse.json(
          category
            ? ROSTER.filter((t) => t.categories?.includes(category))
            : ROSTER,
        );
      }),
      http.get("/api/proxy/api/store/agents", () =>
        HttpResponse.json({
          agents: [],
          pagination: {
            current_page: 1,
            total_items: 0,
            total_pages: 0,
            page_size: 20,
          },
        }),
      ),
      http.get("/api/proxy/api/store/creators", () =>
        HttpResponse.json({
          creators: [],
          pagination: {
            current_page: 1,
            total_items: 0,
            total_pages: 0,
            page_size: 20,
          },
        }),
      ),
    );
  });

  test("the chip row sits above the experts shelf it narrows", async () => {
    render(<MainMarkeplacePage />);

    const chips = await screen.findByRole("group", {
      name: "Browse by category",
    });
    const shelf = await screen.findByRole("link", { name: /Maria/ });

    // Node.DOCUMENT_POSITION_FOLLOWING: the shelf comes after the chips.
    expect(chips.compareDocumentPosition(shelf) & 4).toBeTruthy();
  });

  test("picking a category narrows the roster and drops the others", async () => {
    render(<MainMarkeplacePage />);

    expect(await screen.findByRole("link", { name: /Maria/ })).toBeDefined();
    expect(await screen.findByRole("link", { name: /Max/ })).toBeDefined();

    await userEvent.click(await findCategoryChip("Marketing"));

    await waitFor(() => expect(requestedCategories).toContain("marketing"));
    await waitFor(() =>
      expect(screen.queryByRole("link", { name: /Max/ })).toBeNull(),
    );
    expect(screen.getByRole("link", { name: /Maria/ })).toBeDefined();
  });

  test("clearing the category restores the whole roster", async () => {
    render(<MainMarkeplacePage />);
    await screen.findByRole("link", { name: /Maria/ });

    await userEvent.click(await findCategoryChip("Marketing"));
    await waitFor(() =>
      expect(screen.queryByRole("link", { name: /Max/ })).toBeNull(),
    );

    await userEvent.click(await findCategoryChip("Marketing"));

    expect(await screen.findByRole("link", { name: /Max/ })).toBeDefined();
  });

  test("a category with no experts hides the shelf entirely", async () => {
    server.use(
      http.get("/api/proxy/api/experts/templates", ({ request }) => {
        const category = new URL(request.url).searchParams.get("category");
        return HttpResponse.json(category ? [] : ROSTER);
      }),
    );
    render(<MainMarkeplacePage />);
    await screen.findByRole("link", { name: /Maria/ });

    await userEvent.click(await findCategoryChip("Sales"));

    await waitFor(() =>
      expect(screen.queryByText("Meet the AI Experts")).toBeNull(),
    );
    // Not the empty-roster fallback either: a filtered shelf offers no
    // invitation to raise an expert.
    expect(screen.queryByRole("link", { name: "Raise your own" })).toBeNull();
  });

  test("a failed roster request under a category still offers the raise link", async () => {
    server.use(
      http.get("/api/proxy/api/experts/templates", ({ request }) => {
        const category = new URL(request.url).searchParams.get("category");
        // Unfiltered succeeds so the page renders; the filtered fetch fails.
        return category
          ? new HttpResponse(null, { status: 500 })
          : HttpResponse.json(ROSTER);
      }),
    );
    render(<MainMarkeplacePage />);
    await screen.findByRole("link", { name: /Maria/ });

    await userEvent.click(await findCategoryChip("Sales"));

    // The header carries its own "Raise your own" link, so wait for it to go:
    // a failure is not an answer about the category, so the fallback stays.
    await waitFor(() =>
      expect(screen.queryByText("Meet the AI Experts")).toBeNull(),
    );
    expect(
      screen.getByRole("link", { name: "Raise your own" }).getAttribute("href"),
    ).toBe("/raise");
  });
});

async function findCategoryChip(name: string) {
  const group = await screen.findByRole("group", {
    name: "Browse by category",
  });
  return within(group).findByRole("button", { name });
}
