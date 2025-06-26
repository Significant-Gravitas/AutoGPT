import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { getUsersTransactionHistory } from "@/app/(platform)/admin/spending/actions";
import { CreditTransactionType } from "@/lib/autogpt-server-api";
import { PaginationControls } from "../../../../../../components/ui/pagination-controls";
import { AdminAddMoneyButton } from "../AdminAddMoneyButton";
import { SearchAndFilterFormSpending } from "../SearchAndFilterFormSpending";
import { formatAmount, formatDate, formatType } from "./helpers";

interface Props {
  initialPage?: number;
  initialStatus?: CreditTransactionType;
  initialSearch?: string;
}

export async function AdminUserGrantHistory({
  initialPage = 1,
  initialStatus,
  initialSearch,
}: Props) {
  const { history, pagination } = await getUsersTransactionHistory(
    initialPage,
    15,
    initialSearch,
    initialStatus,
  );

  return (
    <div className="space-y-4">
      <SearchAndFilterFormSpending
        initialStatus={initialStatus}
        initialSearch={initialSearch}
      />

      <div className="rounded-md border bg-white">
        <Table>
          <TableHeader className="bg-gray-50">
            <TableRow>
              <TableHead className="font-medium">User</TableHead>
              <TableHead className="font-medium">Type</TableHead>
              <TableHead className="font-medium">Date</TableHead>
              <TableHead className="font-medium">Reason</TableHead>
              <TableHead className="font-medium">Admin</TableHead>
              <TableHead className="font-medium">Starting Balance</TableHead>
              <TableHead className="font-medium">Amount</TableHead>
              <TableHead className="font-medium">Ending Balance</TableHead>
              {/* <TableHead className="font-medium">Current Balance</TableHead> */}
              <TableHead className="text-right font-medium">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={8}
                  className="py-10 text-center text-gray-500"
                >
                  No transactions found
                </TableCell>
              </TableRow>
            ) : (
              history.map((transaction) => (
                <TableRow
                  key={transaction.user_id}
                  className="hover:bg-gray-50"
                >
                  <TableCell className="font-medium">
                    {transaction.user_email}
                  </TableCell>

                  <TableCell>
                    {formatType(transaction.transaction_type)}
                  </TableCell>
                  <TableCell className="text-gray-600">
                    {formatDate(transaction.transaction_time)}
                  </TableCell>
                  <TableCell>{transaction.reason}</TableCell>
                  <TableCell className="text-gray-600">
                    {transaction.admin_email}
                  </TableCell>
                  <TableCell className="font-medium text-green-600">
                    ${(transaction.running_balance + -transaction.amount) / 100}
                  </TableCell>
                  <TableCell>
                    {formatAmount(
                      transaction.amount,
                      transaction.transaction_type,
                    )}
                  </TableCell>
                  <TableCell className="font-medium text-green-600">
                    ${transaction.running_balance / 100}
                  </TableCell>
                  <TableCell className="text-right">
                    <AdminAddMoneyButton
                      userId={transaction.user_id}
                      userEmail={
                        transaction.user_email ?? "User Email wasn't attached"
                      }
                      currentBalance={transaction.current_balance}
                      defaultAmount={
                        transaction.transaction_type ===
                        CreditTransactionType.USAGE
                          ? -transaction.amount
                          : undefined
                      }
                      defaultComments={
                        transaction.transaction_type ===
                        CreditTransactionType.USAGE
                          ? "Refund for usage"
                          : undefined
                      }
                    />
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
