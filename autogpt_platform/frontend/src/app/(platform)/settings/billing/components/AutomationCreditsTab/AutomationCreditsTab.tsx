"use client";

import { AutoRefillCard } from "./AutoRefillCard/AutoRefillCard";
import { BalanceCard } from "./BalanceCard/BalanceCard";
import { OrgCreditsManagedNotice } from "./OrgCreditsManagedNotice/OrgCreditsManagedNotice";
import { TransactionHistoryCard } from "./TransactionHistoryCard/TransactionHistoryCard";
import { UsageCard } from "./UsageCard/UsageCard";
import { useAutomationCreditsBalance } from "./useAutomationCreditsBalance";

export function AutomationCreditsTab() {
  const { isTeamOrg, isForbidden, balanceCents } =
    useAutomationCreditsBalance();

  if (isForbidden) return <OrgCreditsManagedNotice />;

  if (!isTeamOrg) {
    return (
      <div className="flex flex-col gap-6">
        <BalanceCard index={0} />
        <AutoRefillCard index={1} />
        <UsageCard index={2} />
        <TransactionHistoryCard index={3} />
      </div>
    );
  }

  // A team organization's credits are added by AutoGPT, so there is no
  // auto-refill; its usage and history load once the wallet answers, so a
  // member without billing rights never fires their requests.
  return (
    <div className="flex flex-col gap-6">
      <BalanceCard index={0} />
      {balanceCents !== null ? (
        <>
          <UsageCard index={1} />
          <TransactionHistoryCard index={2} />
        </>
      ) : null}
    </div>
  );
}
