import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2GetSpecificAgentResponseMock,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { server } from "@/mocks/mock-server";
import { hydrateAcrossClocks } from "@/tests/integrations/hydration";
import { cleanup, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import MarketplaceAgentPage from "../page";

// The session only resolves in the browser, after hydration.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isLoggedIn: false, isUserLoading: true }),
}));

const SERVER_NOW = new Date("2026-09-08T12:00:00.000Z");
// The route crawl's visitor: a clock two days ahead of the server's.
const BROWSER_NOW = new Date("2026-09-10T12:00:00.000Z");

const params = { creator: "autogpt", slug: "business-ownerceo-finder" };

describe("marketplace agent page, server-rendered", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    getQueryClient().clear();
    server.use(
      getGetV2GetSpecificAgentMockHandler(
        getGetV2GetSpecificAgentResponseMock({
          agent_name: "Business Owner/CEO Finder",
          creator: "AutoGPT",
          slug: params.slug,
          versions: ["1"],
          agent_image: [],
          agent_video: "",
          agent_output_demo: "",
          changelog: null,
          // Five minutes before the server renders the page.
          last_updated: new Date("2026-09-08T11:55:00.000Z"),
        }),
      ),
      getGetV2ListStoreAgentsMockHandler(
        getGetV2ListStoreAgentsResponseMock({ agents: [] }),
      ),
    );
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  async function loadPage() {
    vi.setSystemTime(SERVER_NOW);
    const page = await MarketplaceAgentPage({
      params: Promise.resolve(params),
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

    expect(serverText).toContain("Last updated Sep 8, 2026");
    expect(screen.getByText("Last updated 2d ago")).toBeDefined();
  });
});
