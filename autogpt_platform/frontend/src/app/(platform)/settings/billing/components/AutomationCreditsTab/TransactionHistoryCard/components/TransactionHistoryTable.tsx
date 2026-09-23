import { Button } from "@/components/atoms/Button/Button";
import { useId, type MouseEvent } from "react";
import { Text } from "@/components/atoms/Text/Text";
import type { Transaction } from "../helpers";
import { useReceiptSelection } from "../useReceiptSelection";
import { TransactionRow } from "./TransactionRow";

type Props = {
  transactions: Transaction[];
  hasMore: boolean;
  isLoadingMore: boolean;
  isLoadMoreError: boolean;
  onLoadMore: () => Promise<unknown>;
};

export function TransactionHistoryTable({
  transactions,
  hasMore,
  isLoadingMore,
  isLoadMoreError,
  onLoadMore,
}: Props) {
  const { selectedID, toggleReceipt, selectRelated } =
    useReceiptSelection(transactions);
  const loadMoreID = useId();
  const endID = useId();

  async function loadOlder(event: MouseEvent<HTMLButtonElement>) {
    const restoreFocus = document.activeElement === event.currentTarget;
    await onLoadMore();
    if (restoreFocus)
      requestAnimationFrame(() => {
        const target =
          document.getElementById(loadMoreID) || document.getElementById(endID);
        target?.focus({ preventScroll: true });
      });
  }
  return (
    <div className="overflow-hidden rounded-large border border-zinc-200 bg-white shadow-subtle">
      <table
        className="w-full table-fixed border-collapse text-left"
        aria-label="Automation credit transactions"
      >
        <colgroup>
          <col />
          <col className="w-0 sm:w-32" />
          <col className="w-24" />
          <col className="w-12 sm:w-24" />
        </colgroup>
        <thead className="border-b border-zinc-200 bg-zinc-50">
          <tr>
            <th scope="col" className="py-3 pl-3 pr-2 sm:pl-5">
              <Text variant="small-medium" as="span" className="text-zinc-600">
                Activity
              </Text>
            </th>
            <th scope="col" className="p-0 sm:px-2 sm:py-3">
              <Text
                variant="small-medium"
                as="span"
                className="sr-only text-zinc-600 sm:not-sr-only"
              >
                Last activity
              </Text>
            </th>
            <th scope="col" className="px-2 py-3 text-right">
              <Text variant="small-medium" as="span" className="text-zinc-600">
                Amount
              </Text>
            </th>
            <th scope="col">
              <span className="sr-only">Details</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((transaction) => (
            <TransactionRow
              key={transaction.id}
              transaction={transaction}
              open={selectedID === transaction.id}
              onToggle={toggleReceipt}
              loadedTransactions={transactions}
              onSelectRelated={selectRelated}
            />
          ))}
        </tbody>
      </table>
      <div className="flex flex-wrap items-center justify-between gap-3 px-3 py-3 sm:px-5">
        <Text variant="small" className="text-zinc-600">
          Latest charges and credits first
        </Text>
        {hasMore ? (
          <Button
            id={loadMoreID}
            variant="secondary"
            size="small"
            className="min-h-11 min-w-0"
            loading={isLoadingMore}
            onClick={(event) => void loadOlder(event)}
          >
            {isLoadMoreError ? "Retry loading more" : "Load more"}
          </Button>
        ) : (
          <Text
            variant="small"
            id={endID}
            tabIndex={-1}
            className="text-zinc-600 focus:outline-none"
          >
            End of history
          </Text>
        )}
        {isLoadMoreError && (
          <Text variant="small" role="alert" className="w-full text-zinc-600">
            We couldn’t load older transactions. Your loaded history is still
            available.
          </Text>
        )}
      </div>
      <Text variant="small" className="sr-only" role="status">
        {isLoadingMore
          ? "Loading older transactions"
          : `${transactions.length} transactions loaded`}
      </Text>
    </div>
  );
}
