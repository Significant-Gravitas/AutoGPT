import { getGetV2GetAdminListingsHistoryMockHandler200 } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import type { StoreSubmissionAdminView } from "@/app/api/__generated__/models/storeSubmissionAdminView";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { server } from "@/mocks/mock-server";
import { hydrateAcrossClocks } from "@/tests/integrations/hydration";
import { screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { AdminAgentsDataTable } from "../components/AdminAgentsDataTable";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/admin/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const SUBMITTED_AT = new Date("2026-11-23T10:00:00.000Z");
// The route crawl opens the page about 25 s after it submits a listing...
const SERVER_NOW = new Date("2026-11-23T10:00:25.000Z");
// ...with the admin's clock five minutes ahead of the server's.
const BROWSER_NOW = new Date("2026-11-23T10:05:25.000Z");

const submission: StoreSubmissionAdminView = {
  listing_id: "listing-1",
  user_id: "user-1",
  slug: "weekly-report",
  listing_version_id: "version-1",
  listing_version: 1,
  graph_id: "graph-1",
  graph_version: 1,
  name: "Weekly report",
  sub_heading: "Drafts the Monday status report",
  description: "",
  instructions: null,
  categories: [],
  image_urls: [],
  video_url: null,
  agent_output_demo_url: null,
  submitted_at: SUBMITTED_AT,
  changes_summary: null,
  status: SubmissionStatus.PENDING,
  internal_comments: null,
};

describe("admin Marketplace page, server-rendered", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    server.use(
      getGetV2GetAdminListingsHistoryMockHandler200({
        listings: [
          {
            listing_id: "listing-1",
            graph_id: "graph-1",
            slug: "weekly-report",
            creator_email: "creator@example.com",
            latest_version: submission,
            versions: [submission],
          },
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 10,
        },
      }),
    );
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  async function loadPage() {
    vi.setSystemTime(SERVER_NOW);
    // A server component: Next awaits it and sends the HTML of what it returns.
    const table = await AdminAgentsDataTable({ initialPage: 1 });
    return hydrateAcrossClocks({
      ui: table,
      serverNow: SERVER_NOW,
      browserNow: BROWSER_NOW,
    });
  }

  test("hydrates without a mismatch when the admin's clock is five minutes ahead", async () => {
    const { hydrationErrors } = await loadPage();

    expect(hydrationErrors).toEqual([]);
  });

  test("serves the submission's date, then how long ago it was by the admin's clock", async () => {
    const { serverText } = await loadPage();

    expect(serverText).toContain("Nov 23, 2026");
    expect(screen.getByText("5 minutes ago")).toBeDefined();
  });
});
