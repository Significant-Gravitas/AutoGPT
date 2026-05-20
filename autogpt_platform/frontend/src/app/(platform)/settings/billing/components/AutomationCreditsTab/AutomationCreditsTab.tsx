"use client";

import { AutoRefillCard } from "./AutoRefillCard/AutoRefillCard";
import { BalanceCard } from "./BalanceCard/BalanceCard";
import { TransactionHistoryCard } from "./TransactionHistoryCard/TransactionHistoryCard";
import { UsageCard } from "./UsageCard/UsageCard";

export function AutomationCreditsTab() {
  return (
    <div className="flex flex-col gap-6">
      <BalanceCard index={0} />
      <AutoRefillCard index={1} />
      <UsageCard index={2} />
      <TransactionHistoryCard index={3} />
    </div>
  );
}
