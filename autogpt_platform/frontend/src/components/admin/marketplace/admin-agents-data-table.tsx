import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  StoreSubmission,
  SubmissionStatus,
} from "@/lib/autogpt-server-api/types";
import { PaginationControls } from "../../ui/pagination-controls";
import { getAdminListingsWithVersions } from "@/app/(platform)/admin/marketplace/actions";
import { ExpandableRow } from "./expandable-row";
import { SearchAndFilterAdminMarketplace } from "./search-filter-form";

// Helper function to get the latest version by version number
const getLatestVersionByNumber = (
  versions: StoreSubmission[],
): StoreSubmission | null => {
  if (!versions || versions.length === 0) return null;
  return versions.reduce(
    (latest, current) =>
      (current.version ?? 0) > (latest.version ?? 1) ? current : latest,
    versions[0],
  );
};

export async function AdminAgentsDataTable({
  initialPage = 1,
  initialStatus,
  initialSearch,
}: {
  initialPage?: number;
  initialStatus?: SubmissionStatus;
  initialSearch?: string;
}) {
  // Server-side data fetching
  const { listings, pagination } = await getAdminListingsWithVersions(
    initialStatus,
    initialSearch,
    initialPage,
    10,
  );

  return (
    <div className="space-y-4">
      <SearchAndFilterAdminMarketplace
        initialStatus={initialStatus}
        initialSearch={initialSearch}
      />

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10"></TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Creator</TableHead>
              <TableHead>Description</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Submitted</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {listings.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center">
                  No submissions found
                </TableCell>
              </TableRow>
            ) : (
              listings.map((listing) => {
                const latestVersion = getLatestVersionByNumber(
                  listing.versions,
                );

                return (
                  <ExpandableRow
                    key={listing.listing_id}
                    listing={listing}
                    latestVersion={latestVersion}
                  />
                );
              })
            )}
          </TableBody>
        </Table>
      </div>

      <PaginationControls
        currentPage={initialPage}
        totalPages={pagination.total_pages}
      />
    </div>
  );
}
