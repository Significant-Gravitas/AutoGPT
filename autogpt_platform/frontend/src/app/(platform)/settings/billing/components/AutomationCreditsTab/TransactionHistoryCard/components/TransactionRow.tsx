import { useId } from "react";
import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  CreditCardIcon,
  Robot01Icon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  activityDescription,
  activityName,
  formatActivityDate,
  isRunActive,
  libraryHref,
  statusLabel,
  type Transaction,
} from "../helpers";
import { TransactionAmount } from "./TransactionAmount";
import { TransactionReceipt } from "./TransactionReceipt";

type Props = {
  transaction: Transaction;
  open: boolean;
  onToggle: (id: string) => void;
  loadedTransactions: Transaction[];
  onSelectRelated: (executionID: string) => void;
};

export function TransactionRow({
  transaction,
  open,
  onToggle,
  loadedTransactions,
  onSelectRelated,
}: Props) {
  const receiptID = useId();
  const posted = formatActivityDate(transaction.transaction_time);
  return (
    <>
      <tr className="border-b border-zinc-100">
        <td className="py-4 pl-3 pr-2 sm:pl-5">
          <TransactionActivity transaction={transaction} />
        </td>
        <td className="p-0 sm:px-2 sm:py-4">
          <Text
            variant="small"
            title={posted.full}
            unmask={false}
            className="hidden sm:block"
          >
            {posted.date}
            <span className="mt-1 block text-zinc-600">{posted.time}</span>
          </Text>
        </td>
        <td className="px-2 py-4 text-right">
          <TransactionAmount amount={transaction.amount} />
          {isRunActive(transaction) && (
            <Text variant="small" className="text-zinc-600">
              so far
            </Text>
          )}
        </td>
        <td className="px-1 py-3 sm:pr-3">
          <Button
            id={`transaction-details-${transaction.id}`}
            variant="ghost"
            size="small"
            unmask={false}
            className="min-h-11 w-full min-w-0 px-1 text-zinc-600"
            onClick={() => onToggle(transaction.id)}
            aria-label={`${open ? "Close" : "Open"} details for ${activityName(transaction)}, ${posted.date} at ${posted.time}`}
            aria-expanded={open}
            aria-controls={open ? receiptID : undefined}
            rightIcon={
              <Icon icon={open ? ArrowUp01Icon : ArrowDown01Icon} size={16} />
            }
          >
            <span className="hidden sm:inline">Details</span>
          </Button>
        </td>
      </tr>
      {open && (
        <tr className="border-b border-zinc-100">
          <td colSpan={4} className="p-0">
            <TransactionReceipt
              transaction={transaction}
              receiptID={receiptID}
              loadedTransactions={loadedTransactions}
              onSelectRelated={onSelectRelated}
            />
          </td>
        </tr>
      )}
    </>
  );
}

function TransactionActivity({ transaction }: { transaction: Transaction }) {
  const href = libraryHref(transaction.library_agent_id);
  const name = activityName(transaction);
  const posted = formatActivityDate(transaction.transaction_time);
  const agent = transaction.activity_type === "agent_run";
  return (
    <div className="flex items-center gap-3">
      <span
        className={cn(
          "hidden size-8 shrink-0 items-center justify-center rounded-small sm:flex",
          agent ? "bg-purple-50 text-purple-700" : "bg-zinc-100 text-zinc-600",
        )}
      >
        <Icon icon={agent ? Robot01Icon : CreditCardIcon} size={16} />
      </span>
      <div className="min-w-0">
        {href ? (
          <Link
            href={href}
            className="inline-flex min-h-6 items-center break-words text-purple-700"
          >
            {name}
          </Link>
        ) : (
          <Text variant="body-medium" unmask={false} className="break-words">
            {name}
          </Text>
        )}
        <Text variant="small" unmask={false} className="mt-1 text-zinc-600">
          {isRunActive(transaction)
            ? statusLabel(transaction.execution_status || "")
            : activityDescription(transaction)}
        </Text>
        <Text
          variant="small"
          unmask={false}
          title={posted.full}
          className="mt-2 text-zinc-600 sm:hidden"
        >
          {posted.date} · {posted.time}
        </Text>
      </div>
    </div>
  );
}
