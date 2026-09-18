import { Text } from "@/components/atoms/Text/Text";
import { activityName, isRunActive, type Transaction } from "../helpers";
import { TransactionAmount } from "./TransactionAmount";

type Props = { transaction: Transaction };

export function ReceiptBreakdown({ transaction }: Props) {
  const usage =
    transaction.activity_type !== "other" &&
    transaction.transaction_type === "USAGE";
  return (
    <dl className="mt-4 grid grid-cols-[minmax(0,1fr)_auto] gap-x-5 gap-y-2">
      {usage ? (
        <>
          <BreakdownLine
            label="Block usage"
            amount={transaction.usage_charge_amount ?? 0}
          />
          {!!transaction.usage_fee_amount && (
            <BreakdownLine
              label="Execution fees"
              amount={transaction.usage_fee_amount}
            />
          )}
          {!!transaction.usage_adjustment_amount && (
            <BreakdownLine
              label="Usage adjustment"
              amount={transaction.usage_adjustment_amount}
            />
          )}
        </>
      ) : (
        <BreakdownLine
          label={activityName(transaction)}
          amount={transaction.amount}
        />
      )}
      <dt className="mt-1 border-t border-zinc-200 pt-3">
        <Text variant="body-medium" as="span">
          {isRunActive(transaction) ? "Net change so far" : "Net change"}
        </Text>
      </dt>
      <dd className="mt-1 border-t border-zinc-200 pt-3 text-right">
        <TransactionAmount amount={transaction.amount} />
      </dd>
    </dl>
  );
}

function BreakdownLine({ label, amount }: { label: string; amount: number }) {
  return (
    <>
      <dt>
        <Text variant="small" as="span" className="text-zinc-600">
          {label}
        </Text>
      </dt>
      <dd className="text-right">
        <TransactionAmount amount={amount} className="text-xs font-normal" />
      </dd>
    </>
  );
}
