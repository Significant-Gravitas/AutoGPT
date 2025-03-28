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
import { getUsersTransactionHistory } from "@/app/admin/spending/actions";
import { AddCreditButton } from "./add-credit-button";
import { CreditTransactionType } from "@/lib/autogpt-server-api";

export async function AdminUserGrantHistory({
  initialPage = 1,
  initialStatus,
  initialSearch,
}: {
  initialPage?: number;
  initialStatus?: CreditTransactionType;
  initialSearch?: string;
}) {
  // Server-side data fetching
  const { history, pagination } = await getUsersTransactionHistory(
    initialPage,
    15,
    initialSearch,
    initialStatus,
  );

  return (
    <div className="space-y-4">
      <SearchAndFilterAdminSpending
        initialStatus={initialStatus}
        initialSearch={initialSearch}
      />

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Email</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Reason</TableHead>
              <TableHead>Admin</TableHead>
              <TableHead>Current Balance</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center">
                  No submissions found
                </TableCell>
              </TableRow>
            ) : (
              history.map((history) => (
                <TableRow key={history.user_id}>
                  <TableCell className="font-medium">
                    {history.user_email}
                  </TableCell>
                  <TableCell>{history.amount}</TableCell>
                  <TableCell>{history.type}</TableCell>
                  <TableCell>{history.date.toString()}</TableCell>
                  <TableCell>{history.reason}</TableCell>
                  <TableCell>{history.admin_email}</TableCell>
                  <TableCell>{history.current_balance}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <AddCreditButton
                        userId={history.user_id}
                        userEmail={history.user_email}
                        currentBalance={history.current_balance}
                        defaultAmount={
                          history.type === CreditTransactionType.USAGE
                            ? -history.amount
                            : undefined
                        }
                        defaultComments={
                          history.type === CreditTransactionType.USAGE
                            ? "Refund for usage"
                            : undefined
                        }
                      />
                    </div>
                  </TableCell>
                </TableRow>
              ))
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
