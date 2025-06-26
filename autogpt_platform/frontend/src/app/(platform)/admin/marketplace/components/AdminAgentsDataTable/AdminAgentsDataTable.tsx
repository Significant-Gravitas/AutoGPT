import { getAdminListingsWithVersions } from "@/app/(platform)/admin/marketplace/actions";
import { PaginationControls } from "@/components/ui/pagination-controls";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { SubmissionStatus } from "@/lib/autogpt-server-api/types";
import { ExpandableRow } from "../ExpandableRow/ExpandableRow";
import { SearchAndFilterForm } from "../SearchAndFilterForm";
import { getLatestVersionByNumber } from "./helpers";

interface Props {
  initialPage?: number;
  initialStatus?: SubmissionStatus;
  initialSearch?: string;
}

export async function AdminAgentsDataTable({
  initialPage = 1,
  initialStatus,
  initialSearch,
}: Props) {
  const { listings, pagination } = await getAdminListingsWithVersions(
    initialStatus,
    initialSearch,
    initialPage,
    10,
  );

  return (
    <div className="space-y-4">
      <SearchAndFilterForm initialSearch={initialSearch} />

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
