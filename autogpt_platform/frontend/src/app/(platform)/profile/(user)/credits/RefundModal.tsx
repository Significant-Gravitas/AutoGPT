import { useState } from "react";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";

import { CreditTransaction } from "@/lib/autogpt-server-api";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  transactions: CreditTransaction[];
  formatCredits: (credit: number) => string;
  refundCredits: (transaction_key: string, reason: string) => Promise<void>;
}

export function RefundModal({
  isOpen,
  onClose,
  transactions,
  formatCredits,
  refundCredits,
}: Props) {
  const [selectedTransactionId, setSelectedTransactionId] =
    useState<string>("");
  const [refundReason, setRefundReason] = useState("");
  const [error, setError] = useState<string | null>(null);

  function handleClose() {
    setSelectedTransactionId("");
    setRefundReason("");
    setError(null);
    onClose();
  }

  function handleRefundRequest() {
    setError(null);

    const selectedTransaction = transactions.find(
      (t) => t.transaction_key === selectedTransactionId,
    );

    if (!selectedTransaction) {
      setError("Please select a transaction to refund");
      return;
    }

    if (refundReason.trim().length < 20) {
      setError("Please provide a clear reason for the refund");
      return;
    }

    refundCredits(selectedTransactionId, refundReason).finally(() =>
      handleClose(),
    );
  }

  const transactionOptions = transactions.map((transaction) => ({
    value: transaction.transaction_key,
    label: `${new Date(transaction.transaction_time).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "numeric",
    })} - ${formatCredits(transaction.amount)}`,
  }));

  return (
    <Dialog
      title="Request Refund"
      styling={{ maxWidth: "425px" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) handleClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="py-4">
          <div className="space-y-4">
            {error && (
              <div className="flex items-center gap-2 rounded-md border border-destructive bg-destructive/10 p-3 text-destructive">
                <Icon icon={AlertCircleIcon} size={16} />
                <Text variant="body" className="text-destructive">
                  {error}
                </Text>
              </div>
            )}

            {transactions.length === 0 ? (
              <Text variant="body" tone="muted">
                No eligible transactions found for refund.
              </Text>
            ) : (
              <Select
                id="refundTransaction"
                label="Select Transaction"
                labelVariant="body-medium"
                placeholder="Select a transaction"
                value={selectedTransactionId}
                onValueChange={setSelectedTransactionId}
                options={transactionOptions}
              />
            )}

            <Input
              id="refundReason"
              type="textarea"
              label="Reason for Refund"
              labelVariant="body-medium"
              placeholder="Please explain why you're requesting a refund..."
              value={refundReason}
              onChange={(e) => setRefundReason(e.target.value)}
              className="min-h-[100px]"
            />
          </div>
        </div>
        <Dialog.Footer>
          <Button variant="secondary" size="md" onClick={handleClose}>
            Cancel
          </Button>
          <Button size="md" onClick={handleRefundRequest}>
            Request Refund
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
