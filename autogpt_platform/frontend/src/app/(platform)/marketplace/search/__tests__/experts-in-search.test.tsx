import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { HttpResponse, delay, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { MainSearchResultPage } from "../components/MainSearchResultPage/MainSearchResultPage";

// These pages wait on several queries before anything renders, and CI is
// slower than a dev machine; the testing-library default is one second.
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
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: false, ready: true }),
    useGetFlag: (flag: string) =>
      flag === "hire-experts"
        ? hireExpertsFlag.enabled
        : actual.useGetFlag(flag as never),
  };
});

const maria: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing",
  bio: null,
  skills: [],
  categories: ["marketing"],
  tagline: "Writes your LinkedIn posts and SEO articles.",
  identity: "You are Maria.",
  voice_preferences: "Direct.",
  boundaries: "Ask first.",
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

/** Every ?search_query= the roster is asked for. */
const searchedTerms: (string | null)[] = [];

function rosterHandler(matches: Expert[]) {
  return http.get("/api/proxy/api/experts/templates", ({ request }) => {
    searchedTerms.push(new URL(request.url).searchParams.get("search_query"));
    return HttpResponse.json(matches);
  });
}

describe("Experts in marketplace search", () => {
  beforeEach(() => {
    searchedTerms.length = 0;
    hireExpertsFlag.enabled = true;
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(),
      getGetV2ListStoreCreatorsMockHandler(),
      getListExpertsMockHandler([]),
    );
  });

  test("returns matching experts alongside agents and creators", async () => {
    server.use(rosterHandler([maria]));

    render(<MainSearchResultPage searchTerm="Maria" sort="runs" />);

    expect(
      await screen.findByRole("heading", { name: "Experts" }),
    ).toBeDefined();
    const card = await screen.findByRole("link", { name: /Maria/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/experts/template-maria",
    );
    expect(screen.getByRole("button", { name: /Experts/ })).toBeDefined();
    await waitFor(() => expect(searchedTerms).toContain("Maria"));
  });

  test("the experts group sits above the workflows", async () => {
    server.use(
      rosterHandler([maria]),
      getGetV2ListStoreAgentsMockHandler({
        agents: [
          {
            slug: "blog-writer",
            agent_name: "Blog Writer",
            agent_image: "",
            creator: "creator",
            creator_avatar: "",
            sub_heading: "",
            description: "",
            runs: 1,
            rating: 5,
            agent_graph_id: "graph-blog-writer",
          },
        ],
        pagination: {
          current_page: 1,
          total_items: 1,
          total_pages: 1,
          page_size: 20,
        },
      }),
    );

    render(<MainSearchResultPage searchTerm="Maria" sort="runs" />);

    const experts = await screen.findByRole("heading", { name: "Experts" });
    // The grid renders a mobile carousel and a desktop grid, so the first
    // card in document order is the one to compare against.
    const [workflow] = await screen.findAllByTestId("store-card");

    // Node.DOCUMENT_POSITION_FOLLOWING: the workflow comes after the heading.
    expect(experts.compareDocumentPosition(workflow) & 4).toBeTruthy();
  });

  test("a search that matches no expert shows no experts group", async () => {
    server.use(rosterHandler([]));

    render(<MainSearchResultPage searchTerm="nothing" sort="runs" />);

    await screen.findByText(/Showing results for/);
    await waitFor(() => expect(searchedTerms).toContain("nothing"));
    expect(screen.queryByRole("heading", { name: "Experts" })).toBeNull();
  });

  test("a roster still in flight shows the skeleton, not 'No results found'", async () => {
    server.use(
      http.get("/api/proxy/api/experts/templates", async () => {
        await delay(3000);
        return HttpResponse.json([maria]);
      }),
      getGetV2ListStoreAgentsMockHandler({
        agents: [],
        pagination: {
          current_page: 1,
          total_items: 0,
          total_pages: 0,
          page_size: 20,
        },
      }),
      getGetV2ListStoreCreatorsMockHandler({
        creators: [],
        pagination: {
          current_page: 1,
          total_items: 0,
          total_pages: 0,
          page_size: 20,
        },
      }),
    );

    render(<MainSearchResultPage searchTerm="Maria" sort="runs" />);

    // Agents and creators resolve empty immediately while the roster is still
    // in flight. `waitFor` cannot catch the resulting flash of the empty state
    // — it retries until the roster lands and the flash is gone — so sample
    // every tick until Maria appears and assert it was never painted.
    let sawEmptyState = false;
    for (let i = 0; i < 80; i++) {
      if (screen.queryByText("No results found")) sawEmptyState = true;
      if (screen.queryByRole("heading", { name: "Experts" })) break;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    expect(sawEmptyState).toBe(false);
    expect(screen.getByRole("heading", { name: "Experts" })).toBeDefined();
  });

  test("asks for no experts and offers no chip outside the beta", async () => {
    hireExpertsFlag.enabled = false;
    server.use(rosterHandler([maria]));

    render(<MainSearchResultPage searchTerm="Maria" sort="runs" />);

    await screen.findByText(/Showing results for/);
    await waitFor(() => expect(searchedTerms).toHaveLength(0));
    expect(screen.queryByRole("heading", { name: "Experts" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Experts/ })).toBeNull();
  });
});
