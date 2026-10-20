"use client";

import type { AccountIpReport } from "@/app/api/__generated__/models/accountIpReport";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { type ReactNode, useState } from "react";
import {
  type BanDuration,
  getBanEnd,
  getErrorMessage,
  pluralize,
} from "../../../helpers";
import { type BanAccountResult, useBanAccounts } from "../../../useBanAccounts";
import type { ChecklistAccount } from "../AccountChecklist";
import { BAN_SENTENCE } from "../BanAllDialog/useBanAllDialog";

interface Args {
  report: AccountIpReport;
  signedInUserId: string | undefined;
  submitBlock: (args: {
    kind: "EMAIL_DOMAIN";
    value: string;
    reason: string;
    expiresAt: Date | null;
    failureTitle: string;
  }) => Promise<boolean>;
  viewBlocksAction: ReactNode;
}

export function useBlockDomainDialog({
  report,
  signedInUserId,
  submitBlock,
  viewBlocksAction,
}: Args) {
  const { toast } = useToast();
  const { banAccounts, isBanning } = useBanAccounts();
  const domain = report.domain.value;
  const accounts: ChecklistAccount[] = report.related
    .filter((account) => account.via_domain && account.id !== signedInUserId)
    .map((account) => ({
      id: account.id,
      email: account.email,
      name: account.name,
      banned: account.banned,
      badges: [],
    }));

  const [isOpen, setIsOpen] = useState(false);
  const [alsoBan, setAlsoBan] = useState(false);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [reason, setReason] = useState("");
  const [duration, setDuration] = useState<BanDuration>("permanent");
  const [isBlocked, setIsBlocked] = useState(false);
  const [results, setResults] = useState<Map<string, BanAccountResult>>(
    new Map(),
  );

  const failedIds = [...results.values()]
    .filter((result) => !result.ok)
    .map((result) => result.userId);
  const toBan = !alsoBan
    ? []
    : failedIds.length > 0
      ? failedIds
      : accounts
          .filter((account) => !account.banned && ticked.has(account.id))
          .map((account) => account.id);

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setAlsoBan(false);
    setTicked(new Set(accounts.filter((a) => !a.banned).map((a) => a.id)));
    setReason("");
    setDuration("permanent");
    setIsBlocked(false);
    setResults(new Map());
  }

  function toggleAccount(id: string) {
    setTicked((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function submit() {
    const expiresAt = getBanEnd(duration);
    if (!isBlocked) {
      const ok = await submitBlock({
        kind: "EMAIL_DOMAIN",
        value: domain,
        reason,
        expiresAt,
        failureTitle: "Couldn't block the domain",
      });
      if (!ok) return;
      setIsBlocked(true);
    }
    if (toBan.length === 0) {
      setIsOpen(false);
      toast({
        title: `Blocked ${domain}`,
        description: "No new sign-ups on it.",
        variant: "success",
        action: viewBlocksAction,
      });
      return;
    }
    let banResults: BanAccountResult[];
    try {
      banResults = await banAccounts({ userIds: toBan, reason, duration });
    } catch (error) {
      toast({
        title: "Couldn't ban the accounts",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      return;
    }
    const merged = new Map(results);
    for (const result of banResults) merged.set(result.userId, result);
    setResults(merged);
    const total = [...merged.values()];
    const failed = total.filter((result) => !result.ok).length;
    if (failed > 0) {
      toast({
        title: `Banned ${total.length - failed} of ${pluralize(total.length, "account")}`,
        description: `${failed} failed; the dialog shows which.`,
        variant: "destructive",
      });
      return;
    }
    setIsOpen(false);
    toast({
      title: `Blocked ${domain} and banned ${pluralize(total.length, "account")}`,
      description: BAN_SENTENCE,
      variant: "success",
      action: viewBlocksAction,
    });
  }

  return {
    domain,
    accounts,
    isOpen,
    handleOpenChange,
    alsoBan,
    setAlsoBan,
    ticked,
    toggleAccount,
    reason,
    setReason,
    duration,
    setDuration,
    results,
    banCount: toBan.length,
    isRetry: failedIds.length > 0,
    isBanning,
    submit,
  };
}
