import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreCategoriesMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { SkillsBrowsePage } from "../components/SkillsBrowsePage/SkillsBrowsePage";

// These pages wait on several queries before anything renders, and CI is
// slower than a dev machine; the testing-library default is one second.
configure({ asyncUtilTimeout: 10000 });

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockPush = vi.hoisted(() => vi.fn());
const mockNotFound = vi.hoisted(() => vi.fn());
const searchParams = vi.hoisted(() => ({ value: "" }));
const flag = vi.hoisted(() => ({ enabled: true, ready: true }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: mockPush,
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/skills",
  useSearchParams: () => new URLSearchParams(searchParams.value),
  useParams: () => ({}),
  notFound: mockNotFound,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useFlagStatus: () => flag };
});

const brandVoice: MarketplaceSkill = {
  slug: "brand-voice-guide",
  name: "brand-voice-guide",
  description: "Write in a consistent brand voice.",
  categories: ["content"],
  required_providers: [],
  install_count: 12,
  creator: null,
  creator_avatar: null,
};

const CATEGORIES = [
  { value: "content", label: "Content", description: "Writing and assets" },
  { value: "finance", label: "Finance", description: "Money and reporting" },
];

function listing(skills: MarketplaceSkill[]) {
  return getGetV2ListMarketplaceSkillsMockHandler200({
    skills,
    pagination: {
      total_items: skills.length,
      total_pages: 1,
      current_page: 1,
      page_size: 20,
    },
  });
}

describe("Marketplace skills browse page", () => {
  beforeEach(() => {
    searchParams.value = "";
    flag.enabled = true;
    flag.ready = true;
    mockPush.mockClear();
    mockNotFound.mockClear();
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getListCopilotSkillsMockHandler200([]),
      getGetV2ListStoreCategoriesMockHandler(CATEGORIES),
    );
  });

  test("lists the catalogue with a count", async () => {
    server.use(listing([brandVoice]));

    render(<SkillsBrowsePage />);

    expect(
      await screen.findByRole("heading", { name: /Otto Skills/ }),
    ).toBeDefined();
    const card = await screen.findByRole("link", { name: /Brand voice guide/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/skills/brand-voice-guide",
    );
    expect(await screen.findByText("1 skill")).toBeDefined();
  });

  test("sends a category choice to the URL", async () => {
    server.use(listing([brandVoice]));

    render(<SkillsBrowsePage />);

    await screen.findByRole("link", { name: /Brand voice guide/ });
    await userEvent.click(screen.getByRole("button", { name: "Content" }));

    expect(mockPush).toHaveBeenCalledWith(
      "/marketplace/skills?category=content",
    );
  });

  test("asks a filtered empty result to show all", async () => {
    searchParams.value = "category=finance";
    server.use(listing([]));

    render(<SkillsBrowsePage />);

    expect(
      await screen.findByTestId("skills-browse-empty", undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
    expect(screen.getByText(/No Finance skills yet/)).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Show all" }));
    expect(mockPush).toHaveBeenCalledWith("/marketplace/skills");
  });

  test("keeps a retry on the page when the catalogue fails", async () => {
    server.use(
      http.get("/api/proxy/api/store/skills", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 500 }),
      ),
    );

    render(<SkillsBrowsePage />);

    expect(
      await screen.findByText(/Couldn't load skills right now/, undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
  });

  test("is not found when the skills-hub flag is off", async () => {
    flag.enabled = false;
    server.use(listing([brandVoice]));

    render(<SkillsBrowsePage />);

    await waitFor(() => expect(mockNotFound).toHaveBeenCalled());
  });
});
