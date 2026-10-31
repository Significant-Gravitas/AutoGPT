import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@/components/molecules/Table/components/TablePrimitives";

import { PaginationControls } from "@/components/molecules/PaginationControls/PaginationControls";
import { SearchAndFilterAdminSpending } from "./SearchAndFilterAdminSpending";
import { getUsersTransactionHistory } from "@/app/(platform)/admin/spending/actions";
import { AdminAddMoneyButton } from "./AddMoneyButton";
import { RateLimitModal } from "./RateLimitModal";
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
      ? "text-success"
      : isNeutral
        ? "text-info-foreground"
        : "text-error-foreground";
    return <span className={color}>${Math.abs(amount / 100)}</span>;
  };

  // Helper function to format the transaction type with color
  const formatType = (type: CreditTransactionType) => {
    const isGrant = type === CreditTransactionType.GRANT;
    const isPurchased = type === CreditTransactionType.TOP_UP;
    const isSpent = type === CreditTransactionType.USAGE;

    const displayText = type;
    let bgColor = "";

    if (isGrant) {
      bgColor = "bg-success/30 text-success-foreground";
    } else if (isPurchased) {
      bgColor = "bg-info/20 text-info-foreground";
    } else if (isSpent) {
      bgColor = "bg-error-surface text-error-foreground";
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
      <SearchAndFilterAdminSpending initialSearch={initialSearch} />

      <div className="rounded-md border bg-surface">
        <Table>
          <thead className="bg-surface-muted [&_tr]:border-b">
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Reason</TableHead>
              <TableHead>Admin</TableHead>
              <TableHead>Starting Balance</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Ending Balance</TableHead>
              {/* <TableHead>Current Balance</TableHead> */}
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </thead>
          <TableBody>
            {history.length === 0 ? (
              <TableRow>
                <td
                  colSpan={9}
                  className="px-2 py-10 text-center align-middle text-foreground-tertiary"
                >
                  No transactions found
                </td>
              </TableRow>
            ) : (
              history.map((transaction) => (
                <TableRow
                  key={`${transaction.user_id}-${transaction.transaction_time}`}
                >
                  <TableCell>
                    <span className="font-medium">
                      {transaction.user_email}
                    </span>
                  </TableCell>

                  <TableCell>
                    {formatType(transaction.transaction_type)}
                  </TableCell>
                  <TableCell>
                    <span className="text-foreground-secondary">
                      {formatDate(transaction.transaction_time)}
                    </span>
                  </TableCell>
                  <TableCell>{transaction.reason}</TableCell>
                  <TableCell>
                    <span className="text-foreground-secondary">
                      {transaction.admin_email}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className="font-medium text-success">
                      $
                      {((transaction.running_balance ?? 0) +
                        -transaction.amount) /
                        100}
                    </span>
                  </TableCell>
                  <TableCell>
                    {formatAmount(
                      transaction.amount,
                      transaction.transaction_type,
                    )}
                  </TableCell>
                  <TableCell>
                    <span className="font-medium text-success">
                      ${(transaction.running_balance ?? 0) / 100}
                    </span>
                  </TableCell>
                  {/* <TableCell className="font-medium text-success">
                    ${transaction.current_balance / 100}
                  </TableCell> */}
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-2">
                      <RateLimitModal
                        userId={transaction.user_id}
                        userEmail={transaction.user_email ?? ""}
                      />
                      <AdminAddMoneyButton
                        userId={transaction.user_id}
                        userEmail={transaction.user_email ?? ""}
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
