import {
  getGetV2GetMarketplaceSkillMockHandler200,
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListMarketplaceSkillsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { server } from "@/mocks/mock-server";
import { hydrateAcrossClocks } from "@/tests/integrations/hydration";
import { cleanup, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import MarketplaceSkillPage from "../page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/marketplace/skills/outreach-playbook",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ slug: "outreach-playbook" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

// The session only resolves in the browser, after hydration.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isLoggedIn: false, isUserLoading: true }),
}));

const SERVER_NOW = new Date("2026-09-08T12:00:00.000Z");
// The route crawl's visitor: a clock two days ahead of the server's.
const BROWSER_NOW = new Date("2026-09-10T12:00:00.000Z");

const outreach = {
  slug: "outreach-playbook",
  name: "outreach-playbook",
  title: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: ["google"],
  install_count: 3,
  creator: null,
  creator_avatar: null,
  skill_listing_version_id: "version-1",
  body: "",
  triggers: [],
  // Five minutes before the server renders the page.
  updated_at: new Date("2026-09-08T11:55:00.000Z"),
} as MarketplaceSkillDetails;

describe("marketplace skill page, server-rendered", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    getQueryClient().clear();
    server.use(
      getGetV2GetMarketplaceSkillMockHandler200(outreach),
      getGetV2ListMarketplaceSkillsMockHandler200(
        getGetV2ListMarketplaceSkillsResponseMock({ skills: [] }),
      ),
    );
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  async function loadPage() {
    vi.setSystemTime(SERVER_NOW);
    const page = await MarketplaceSkillPage({
      params: Promise.resolve({ slug: "outreach-playbook" }),
    });
    return hydrateAcrossClocks({
      ui: page,
      serverNow: SERVER_NOW,
      browserNow: BROWSER_NOW,
    });
  }

  test("hydrates without a mismatch when the visitor's clock is two days ahead", async () => {
    const { hydrationErrors } = await loadPage();

    expect(hydrationErrors).toEqual([]);
  });

  test("serves the update's date, then how long ago it was by the visitor's clock", async () => {
    const { serverText } = await loadPage();

    expect(serverText).toContain("Updated Sep 8, 2026");
    expect(screen.getByText("Updated 2d ago")).toBeDefined();
  });
});
