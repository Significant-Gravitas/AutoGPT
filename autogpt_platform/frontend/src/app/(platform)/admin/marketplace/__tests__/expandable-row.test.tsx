import type { StoreListingWithVersionsAdminView } from "@/app/api/__generated__/models/storeListingWithVersionsAdminView";
import type { StoreSubmissionAdminView } from "@/app/api/__generated__/models/storeSubmissionAdminView";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { ExpandableRow } from "../components/ExpandleRow";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/admin/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

function submission(status: SubmissionStatus): StoreSubmissionAdminView {
  return {
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
    submitted_at: null,
    changes_summary: null,
    status,
    internal_comments: null,
  };
}

function renderRow(status: SubmissionStatus) {
  const latestVersion = submission(status);
  const listing: StoreListingWithVersionsAdminView = {
    listing_id: "listing-1",
    graph_id: "graph-1",
    slug: "weekly-report",
    creator_email: "creator@example.com",
    latest_version: latestVersion,
    versions: [latestVersion],
  };
  render(
    <table>
      <tbody>
        <ExpandableRow listing={listing} latestVersion={latestVersion} />
      </tbody>
    </table>,
  );
}

describe("ExpandableRow status badge", () => {
  test("shows a submission waiting for review as Pending", () => {
    renderRow(SubmissionStatus.PENDING);
    expect(screen.getByText("Pending")).toBeDefined();
  });

  test("shows a rejected submission as Rejected", () => {
    renderRow(SubmissionStatus.REJECTED);
    expect(screen.getByText("Rejected")).toBeDefined();
  });

  test("shows an approved submission as Approved", () => {
    renderRow(SubmissionStatus.APPROVED);
    expect(screen.getByText("Approved")).toBeDefined();
  });

  test("shows a submission that was never sent for review as Draft", () => {
    renderRow(SubmissionStatus.DRAFT);
    expect(screen.getByText("Draft")).toBeDefined();
  });
});
