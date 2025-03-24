
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { PaginationControls } from "../../ui/pagination-controls";
import { SearchAndFilterAdminSpending } from "./search-filter-form";
import { getGrantHistory } from "@/app/admin/spending/actions";


export async function AdminUserGrantHistory({
  initialPage = 1,
  initialSearch,
}: {
  initialPage?: number;
  initialSearch?: string;
}) {
  // Server-side data fetching
  const { grants, pagination } = await getGrantHistory(
    initialPage,
    10,
    initialSearch,
  );

  return (
    <>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10"></TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Reason</TableHead>
              <TableHead>Admin</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {grants.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center">
                  No submissions found
                </TableCell>
              </TableRow>
            ) : (
              grants.map((grant) => (
                <TableRow key={grant.user_id}>
                  <TableCell className="font-medium">
                    {grant.user_email}
                  </TableCell>
                  <TableCell>{grant.amount}</TableCell>
                  <TableCell>{grant.date.toLocaleDateString()}</TableCell>
                  <TableCell>{grant.reason}</TableCell>
                  <TableCell>{grant.admin_email}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <PaginationControls
        currentPage={initialPage}
        totalPages={pagination.total_pages}
        pathParam="grantPage"
      />
    </ >
  );
}
