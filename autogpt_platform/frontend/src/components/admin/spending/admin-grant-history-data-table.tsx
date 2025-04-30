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
import { AdminAddMoneyButton } from "./add-money-button";
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

  // Helper function to format the amount with color based on transaction type
  const formatAmount = (amount: number, type: CreditTransactionType) => {
    const isPositive = type === CreditTransactionType.GRANT;
    const isNeutral = type === CreditTransactionType.TOP_UP;
    const color = isPositive
      ? "text-green-600"
      : isNeutral
        ? "text-blue-600"
        : "text-red-600";
    return <span className={color}>${Math.abs(amount / 100)}</span>;
  };

  // Helper function to format the transaction type with color
  const formatType = (type: CreditTransactionType) => {
    const isGrant = type === CreditTransactionType.GRANT;
    const isPurchased = type === CreditTransactionType.TOP_UP;
    const isSpent = type === CreditTransactionType.USAGE;

    let displayText = type;
    let bgColor = "";

    if (isGrant) {
      bgColor = "bg-green-100 text-green-800";
    } else if (isPurchased) {
      bgColor = "bg-blue-100 text-blue-800";
    } else if (isSpent) {
      bgColor = "bg-red-100 text-red-800";
    }

    return (
      <span className={`rounded-full px-2 py-1 text-xs font-medium ${bgColor}`}>
        {displayText.valueOf()}
      </span>
    );
  };

  // Helper function to format the date
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    }).format(new Date(date));
  };

  return (
    <div className="space-y-4">
      <SearchAndFilterAdminSpending
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
                  {/* <TableCell className="font-medium text-green-600">
                    ${transaction.current_balance / 100}
                  </TableCell> */}
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
