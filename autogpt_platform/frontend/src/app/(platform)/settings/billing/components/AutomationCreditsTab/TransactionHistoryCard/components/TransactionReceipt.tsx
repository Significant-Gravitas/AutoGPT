import { Text } from "@/components/atoms/Text/Text";
import { activityName, receiptNote, type Transaction } from "../helpers";
import { ReceiptBreakdown } from "./ReceiptBreakdown";
import { ReceiptEntries } from "./ReceiptEntries";
import { ReceiptMetadata } from "./ReceiptMetadata";
import { ReceiptRelations } from "./ReceiptRelations";

type Props = {
  transaction: Transaction;
  receiptID: string;
  loadedTransactions: Transaction[];
  onSelectRelated: (executionID: string) => void;
};

export function TransactionReceipt({
  transaction,
  receiptID,
  loadedTransactions,
  onSelectRelated,
}: Props) {
  const hasRelations =
    !!transaction.parent_execution_id ||
    !!transaction.related_executions?.length ||
    !!transaction.related_executions_has_more;
  const heading =
    transaction.activity_type === "agent_run"
      ? `Charges for this run${hasRelations ? " only" : ""}`
      : transaction.activity_type === "copilot_tools"
        ? "Tool usage in this conversation"
        : "Credit balance change";
  const note = receiptNote(transaction);
  return (
    <section
      id={receiptID}
      aria-label={`${activityName(transaction)} credit receipt`}
      className="mx-2 mb-3 rounded-medium border border-zinc-100 bg-zinc-50 p-3 sm:mx-4 sm:p-5"
    >
      <Text variant="body-medium" as="h3">
        {heading}
      </Text>
      <ReceiptBreakdown transaction={transaction} />
      {note && (
        <Text variant="small" className="mt-3 text-zinc-600">
          {note}
        </Text>
      )}
      <ReceiptRelations
        transaction={transaction}
        loadedTransactions={loadedTransactions}
        onSelectRelated={onSelectRelated}
      />
      <ReceiptMetadata transaction={transaction} />
      <ReceiptEntries transaction={transaction} />
    </section>
  );
}
