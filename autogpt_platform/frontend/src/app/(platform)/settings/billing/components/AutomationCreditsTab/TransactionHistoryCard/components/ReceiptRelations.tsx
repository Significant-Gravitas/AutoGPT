import type { CreditHistoryRelatedExecution } from "@/app/api/__generated__/models/creditHistoryRelatedExecution";
import { Button } from "@/components/atoms/Button/Button";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { libraryHref, type Transaction } from "../helpers";
import { TransactionAmount } from "./TransactionAmount";

type Props = {
  transaction: Transaction;
  loadedTransactions: Transaction[];
  onSelectRelated: (executionID: string) => void;
};

export function ReceiptRelations({
  transaction,
  loadedTransactions,
  onSelectRelated,
}: Props) {
  const related = transaction.related_executions ?? [];
  if (
    !transaction.parent_execution_id &&
    related.length === 0 &&
    !transaction.related_executions_has_more
  )
    return null;
  return (
    <div className="mt-4 space-y-2 border-t border-zinc-200 pt-3">
      {transaction.parent_execution_id && (
        <Text variant="small" className="text-zinc-600">
          Started by
        </Text>
      )}
      {transaction.parent_execution_id && (
        <RelatedExecution
          execution={{
            execution_id: transaction.parent_execution_id,
            agent_name: transaction.parent_agent_name,
            library_agent_id: transaction.parent_library_agent_id,
            execution_available: !!transaction.parent_library_agent_id,
          }}
          loadedTransactions={loadedTransactions}
          receiptAsOf={transaction.receipt_as_of}
          onSelectRelated={onSelectRelated}
        />
      )}
      {(related.length > 0 || transaction.related_executions_has_more) && (
        <Text variant="small" className="text-zinc-600">
          Other agents used by this run
        </Text>
      )}
      {related.map((execution) => (
        <RelatedExecution
          key={execution.execution_id}
          execution={execution}
          loadedTransactions={loadedTransactions}
          receiptAsOf={transaction.receipt_as_of}
          onSelectRelated={onSelectRelated}
        />
      ))}
      <Text variant="small" className="text-zinc-600">
        Any credit charges for related runs appear separately.
        {related.length > 0
          ? " The task page includes related runs in its total."
          : ""}
      </Text>
      {transaction.related_executions_has_more && (
        <Text variant="small" className="text-zinc-600">
          More related runs are not shown here.
        </Text>
      )}
    </div>
  );
}

function RelatedExecution({
  execution,
  loadedTransactions,
  receiptAsOf,
  onSelectRelated,
}: {
  execution: CreditHistoryRelatedExecution;
  loadedTransactions: Transaction[];
  receiptAsOf: Transaction["receipt_as_of"];
  onSelectRelated: (executionID: string) => void;
}) {
  const name = execution.agent_name || "Agent unavailable";
  const loaded = loadedTransactions.find(
    (item) => item.usage_execution_id === execution.execution_id,
  );
  const sameSnapshot =
    receiptAsOf &&
    loaded?.receipt_as_of &&
    new Date(receiptAsOf).getTime() ===
      new Date(loaded.receipt_as_of).getTime();
  const amount = sameSnapshot ? loaded.amount : execution.amount;
  const agentHref = libraryHref(execution.library_agent_id);
  const href =
    agentHref && execution.execution_available
      ? `${agentHref}?activeTab=runs&activeItem=${encodeURIComponent(execution.execution_id)}`
      : null;
  return (
    <div className="flex items-center justify-between gap-4">
      {loaded ? (
        <Button
          variant="ghost"
          size="small"
          unmask={false}
          className="min-h-11 min-w-0 whitespace-normal px-0 text-left text-purple-700"
          onClick={() => onSelectRelated(execution.execution_id)}
        >
          {name}
        </Button>
      ) : href ? (
        <Link
          href={href}
          className="inline-flex min-h-11 items-center text-purple-700"
        >
          {name}
        </Link>
      ) : (
        <Text variant="small" unmask={false}>
          {name}
        </Text>
      )}
      {amount != null && (
        <TransactionAmount amount={amount} className="text-xs" />
      )}
    </div>
  );
}
