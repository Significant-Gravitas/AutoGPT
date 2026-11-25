import { getGetV2AdminPreviewSubmissionListingUrl } from "@/app/api/__generated__/endpoints/admin/admin";
import {
  getGetV2AdminPreviewSubmissionListingMockHandler,
  getPostV2AdminAddPendingAgentToLibraryMockHandler,
  getPostV2AdminAddPendingAgentToLibraryResponseMock,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { AdminAgentPreview } from "../components/AdminAgentPreview/AdminAgentPreview";

const VERSION_ID = "version-1";

const goBack = vi.fn();

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: VERSION_ID }),
  useRouter: () => ({
    back: goBack,
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => `/admin/marketplace/preview/${VERSION_ID}`,
  useSearchParams: () => new URLSearchParams(),
}));

function submission(overrides: Partial<StoreAgentDetails> = {}) {
  const listing: StoreAgentDetails = {
    store_listing_version_id: VERSION_ID,
    slug: "weekly-report",
    agent_name: "Weekly report",
    agent_video: "",
    agent_output_demo: "",
    agent_image: [],
    creator: "Ada",
    creator_avatar: "",
    sub_heading: "Drafts the Monday status report",
    description: "Reads last week's tickets and drafts the report.",
    instructions: null,
    categories: ["productivity"],
    runs: 0,
    rating: 0,
    versions: ["1"],
    graph_id: "graph-1",
    graph_versions: ["1"],
    last_updated: new Date("2026-11-20T09:00:00Z"),
    recommended_schedule_cron: null,
    active_version_id: VERSION_ID,
    has_approved_version: false,
    ...overrides,
  };
  return listing;
}

function servePreview(listing: StoreAgentDetails) {
  server.use(getGetV2AdminPreviewSubmissionListingMockHandler(listing));
}

function renderPreview() {
  render(
    <>
      <AdminAgentPreview />
      <Toaster />
    </>,
  );
}

test("an admin sees a submission waiting for review as the Marketplace would list it, marked pending", async () => {
  servePreview(
    submission({
      instructions: "Connect Gmail before the first run.",
      recommended_schedule_cron: "0 9 * * 1",
    }),
  );
  renderPreview();

  expect(
    await screen.findByText("Admin Preview — Pending Approval"),
  ).toBeDefined();
  expect(screen.getAllByText("Weekly report").length).toBeGreaterThan(0);
  expect(screen.getByText("Connect Gmail before the first run.")).toBeDefined();
  expect(screen.getByText("0 9 * * 1")).toBeDefined();
  expect(screen.getByText("weekly-report")).toBeDefined();
  expect(screen.getByText("No images or videos submitted")).toBeDefined();
});

test("a submission with an approved version isn't marked pending", async () => {
  servePreview(submission({ has_approved_version: true }));
  renderPreview();

  expect(await screen.findByText("Admin Preview")).toBeDefined();
  expect(screen.queryByText(/Pending Approval/)).toBeNull();
});

test("adding a submission to the admin's library adds that version and says so", async () => {
  servePreview(submission());
  const added: string[] = [];
  server.use(
    getPostV2AdminAddPendingAgentToLibraryMockHandler(({ params }) => {
      added.push(String(params.storeListingVersionId));
      return getPostV2AdminAddPendingAgentToLibraryResponseMock();
    }),
  );
  renderPreview();

  await userEvent.click(
    await screen.findByRole("button", { name: "Add to My Library" }),
  );

  expect(await screen.findByText("Added to Library")).toBeDefined();
  expect(added).toEqual([VERSION_ID]);
  expect(
    await screen.findByRole("button", { name: "Add to My Library" }),
  ).toBeDefined();
});

test("a submission the backend can't find shows the backend's reason and offers a way back", async () => {
  server.use(
    // The generated handlers answer only with the API's documented codes.
    http.get(`*${getGetV2AdminPreviewSubmissionListingUrl(VERSION_ID)}`, () =>
      HttpResponse.json(
        { detail: "Store listing version not found" },
        { status: 404 },
      ),
    ),
  );
  renderPreview();

  expect(
    await screen.findByText("Store listing version not found"),
  ).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: "Go back" }));
  expect(goBack).toHaveBeenCalledTimes(1);
});
