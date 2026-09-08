import { useEffect, useState } from "react";
import type { Transaction } from "./helpers";

export function useReceiptSelection(transactions: Transaction[]) {
  const [selectedID, setSelectedID] = useState<string | null>(null);
  const [relatedID, setRelatedID] = useState<string | null>(null);

  useEffect(() => {
    if (!relatedID) return;
    const button = document.getElementById(`transaction-details-${relatedID}`);
    button?.focus({ preventScroll: true });
    button?.scrollIntoView({ block: "center" });
    setRelatedID(null);
  }, [relatedID]);

  function toggleReceipt(id: string) {
    setSelectedID((current) => (current === id ? null : id));
  }

  function selectRelated(executionID: string) {
    const transaction = transactions.find(
      (item) => item.usage_execution_id === executionID,
    );
    if (!transaction) return;
    setSelectedID(transaction.id);
    setRelatedID(transaction.id);
  }

  return { selectedID, toggleReceipt, selectRelated };
}
