"use client";

import type { AccountIpReport } from "@/app/api/__generated__/models/accountIpReport";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import {
  type AdminUser,
  type BanDuration,
  getBanEnd,
  getErrorMessage,
  pluralize,
} from "../../../helpers";
import { useAccessBlocks } from "../../../useAccessBlocks";
import { type BanAccountResult, useBanAccounts } from "../../../useBanAccounts";
import type { ChecklistAccount } from "../AccountChecklist";
import { getAlsoBlockOptions, getBanAllAccounts } from "./helpers";

export const BAN_SENTENCE =
  "Their sessions are signed out. Tokens they hold or get in the next 5 minutes work for up to an hour, and their API keys, schedules and triggers keep running.";

interface Args {
  user: AdminUser;
  report: AccountIpReport;
  signedInUserId: string | undefined;
}

export function useBanAllDialog({ user, report, signedInUserId }: Args) {
  const { toast } = useToast();
  const { banAccounts, isBanning } = useBanAccounts();
  const { createBlock, isCreating } = useAccessBlocks();
  const { accounts, excludesSelf } = getBanAllAccounts({
    user,
    report,
    signedInUserId,
  });
  const blockOptions = getAlsoBlockOptions(report);

  const [isOpen, setIsOpen] = useState(false);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [blocksTicked, setBlocksTicked] = useState<Set<string>>(new Set());
  const [reason, setReason] = useState("");
  const [duration, setDuration] = useState<BanDuration>("permanent");
  const [results, setResults] = useState<Map<string, BanAccountResult>>(
    new Map(),
  );
  const [blocksDone, setBlocksDone] = useState<Set<string>>(new Set());

  const failedIds = [...results.values()]
    .filter((result) => !result.ok)
    .map((result) => result.userId);
  const isRetry = failedIds.length > 0;
  const toBan = isRetry ? failedIds : banTargets(accounts, ticked);
  const toBlock = blockOptions.filter(
    (option) => blocksTicked.has(option.key) && !blocksDone.has(option.key),
  );
  const tickedAdmins = accounts.filter(
    (account) => account.role === "admin" && toBan.includes(account.id),
  ).length;

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setTicked(new Set(accounts.filter((a) => !a.banned).map((a) => a.id)));
    setBlocksTicked(new Set());
    setReason("");
    setDuration("permanent");
    setResults(new Map());
    setBlocksDone(new Set());
  }

  function toggle(set: Set<string>, key: string) {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  }

  async function submit() {
    let banResults: BanAccountResult[] = [];
    if (toBan.length > 0) {
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
    }
    const merged = new Map(results);
    for (const result of banResults) merged.set(result.userId, result);
    setResults(merged);

    const blocked = new Set(blocksDone);
    const blockErrors: string[] = [];
    for (const option of toBlock) {
      try {
        await createBlock({
          kind: option.kind,
          value: option.value,
          reason,
          expiresAt: getBanEnd(duration),
        });
        blocked.add(option.key);
      } catch (error) {
        blockErrors.push(`${option.label}: ${getErrorMessage(error)}`);
      }
    }
    setBlocksDone(blocked);

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
    if (blockErrors.length > 0) {
      toast({
        title: "Couldn't block every address",
        description: blockErrors.join(" "),
        variant: "destructive",
      });
      return;
    }
    setIsOpen(false);
    toast({
      title: summarise(total.length, blocked.size),
      description: BAN_SENTENCE,
      variant: "success",
    });
  }

  return {
    isOpen,
    handleOpenChange,
    accounts,
    excludesSelf,
    blockOptions,
    ticked,
    toggleAccount: (id: string) => setTicked((set) => toggle(set, id)),
    blocksTicked,
    toggleBlock: (key: string) => setBlocksTicked((set) => toggle(set, key)),
    blocksDone,
    reason,
    setReason,
    duration,
    setDuration,
    results,
    banCount: toBan.length,
    blockCount: toBlock.length,
    tickedAdmins,
    isRetry,
    isSubmitting: isBanning || isCreating,
    submit,
  };
}

function banTargets(accounts: ChecklistAccount[], ticked: Set<string>) {
  return accounts
    .filter((account) => !account.banned && ticked.has(account.id))
    .map((account) => account.id);
}

function summarise(banned: number, blocked: number) {
  const bans = `Banned ${pluralize(banned, "account")}`;
  return blocked > 0 ? `${bans} and blocked ${blocked}` : bans;
}
