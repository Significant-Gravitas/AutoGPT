import { useId, useState } from "react";
import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { chargeLabel, formatActivityDate, type Transaction } from "../helpers";
import { TransactionAmount } from "./TransactionAmount";

type Props = { transaction: Transaction };

export function ReceiptEntries({ transaction }: Props) {
  const [entriesOpen, setEntriesOpen] = useState(false);
  const [referenceOpen, setReferenceOpen] = useState(false);
  const entriesID = useId();
  const referenceID = useId();
  const reference =
    transaction.usage_execution_id || transaction.transaction_key;
  return (
    <div className="mt-3">
      <div className="flex flex-wrap items-center justify-between gap-x-4">
        {!!transaction.charges?.length && (
          <Button
            variant="ghost"
            size="small"
            className="min-h-11 min-w-0 px-0 text-zinc-600"
            onClick={() => setEntriesOpen(!entriesOpen)}
            aria-expanded={entriesOpen}
            aria-controls={entriesOpen ? entriesID : undefined}
            rightIcon={
              <Icon
                icon={entriesOpen ? ArrowUp01Icon : ArrowDown01Icon}
                size={16}
              />
            }
          >
            {entriesOpen ? "Hide" : "Show"} charge entries
          </Button>
        )}
        {reference && (
          <Button
            variant="ghost"
            size="small"
            className="min-h-11 min-w-0 px-0 text-zinc-600"
            onClick={() => setReferenceOpen(!referenceOpen)}
            aria-expanded={referenceOpen}
            aria-controls={referenceOpen ? referenceID : undefined}
          >
            {referenceOpen ? "Hide reference" : "Reference"}
          </Button>
        )}
      </div>
      {entriesOpen && (
        <div id={entriesID}>
          <ChargeEntries transaction={transaction} />
        </div>
      )}
      {referenceOpen && (
        <div id={referenceID} className="mt-2">
          <Text variant="small" className="text-zinc-600">
            {transaction.activity_type === "agent_run" ? "Run ID" : "Reference"}
          </Text>
          <Text
            variant="small"
            as="code"
            unmask={false}
            className="break-all font-mono"
          >
            {reference}
          </Text>
        </div>
      )}
    </div>
  );
}

function ChargeEntries({ transaction }: Props) {
  const charges = transaction.charges ?? [];
  return (
    <div className="mt-2 border-t border-zinc-200 pt-3">
      <table className="w-full text-left" aria-label="Recorded charge entries">
        <thead className="sr-only">
          <tr>
            <th scope="col">Date and time</th>
            <th scope="col">Charge</th>
            <th scope="col">Amount</th>
          </tr>
        </thead>
        <tbody>
          {charges.map((charge) => {
            const posted = formatActivityDate(charge.posted_at);
            return (
              <tr key={charge.id}>
                <td className="py-2 pr-3 align-top">
                  <Text
                    variant="small"
                    unmask={false}
                    title={posted.full}
                    className="text-zinc-600"
                  >
                    {posted.date}
                    <br />
                    {posted.time}
                  </Text>
                </td>
                <td className="py-2 pr-3 align-top">
                  <Text variant="small">{chargeLabel(charge.charge_type)}</Text>
                </td>
                <td className="py-2 text-right align-top">
                  <TransactionAmount
                    amount={charge.amount}
                    className="text-xs font-normal"
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {transaction.charges_truncated && (
        <Text variant="small" className="mt-2 text-zinc-600">
          Showing {charges.length} of {transaction.charges_total_count} charge
          entries. The total includes every charge and adjustment.
        </Text>
      )}
    </div>
  );
}
