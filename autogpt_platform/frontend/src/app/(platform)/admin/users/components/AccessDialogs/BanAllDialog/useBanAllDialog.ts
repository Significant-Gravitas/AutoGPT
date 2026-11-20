"use client";

import type { AccountIPReport } from "@/app/api/__generated__/models/accountIPReport";
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
import {
  BAN_SENTENCE,
  getBanTargets,
  hasFailedBans,
  mergeBanResults,
  partialBanFailure,
} from "../helpers";
import {
  type AlsoBlockOption,
  describeBlocks,
  getAlsoBlockOptions,
  getBanAllAccounts,
} from "./helpers";

interface Args {
  user: AdminUser;
  report: AccountIPReport;
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

  const toBan = getBanTargets({ accounts, ticked, results });
  const toBlock = blockOptions.filter(
    (option) => blocksTicked.has(option.key) && !blocksDone.has(option.key),
  );
  const isBlockOnly = toBan.length === 0 && toBlock.length > 0;
  const blockSummary = describeBlocks(toBlock);

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setTicked(
      new Set(
        accounts
          .filter((account) => !account.banned && account.isTickedByDefault)
          .map((account) => account.id),
      ),
    );
    setBlocksTicked(new Set());
    setReason("");
    setDuration("permanent");
    setResults(new Map());
    setBlocksDone(new Set());
  }

  async function blockTicked() {
    const blocked = new Set(blocksDone);
    const errors: string[] = [];
    for (const option of toBlock) {
      try {
        await createBlock({
          ...blockArgs(option),
          reason,
          expiresAt: getBanEnd(duration),
        });
        blocked.add(option.key);
      } catch (error) {
        errors.push(`${option.label}: ${getErrorMessage(error)}`);
      }
    }
    setBlocksDone(blocked);
    return { blocked: blocked.size, errors };
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
    const { merged, total, failed } = mergeBanResults(results, banResults);
    setResults(merged);
    const blocks = await blockTicked();
    if (failed > 0) {
      toast(partialBanFailure(total, failed, blocks.errors));
      return;
    }
    if (blocks.errors.length > 0) {
      toast({
        title: "Couldn't block every address",
        description: blocks.errors.join(" "),
        variant: "destructive",
      });
      return;
    }
    setIsOpen(false);
    if (total === 0) {
      toast({ title: `Blocked ${blockSummary}`, variant: "success" });
      return;
    }
    const bans = `Banned ${pluralize(total, "account")}`;
    toast({
      title:
        blocks.blocked > 0 ? `${bans} and blocked ${blocks.blocked}` : bans,
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
    isBlockOnly,
    blockSummary,
    tickedAdmins: accounts.filter(
      (account) => account.role === "admin" && toBan.includes(account.id),
    ).length,
    isRetry: hasFailedBans(results),
    isSubmitting: isBanning || isCreating,
    submit,
  };
}

function toggle(set: Set<string>, key: string) {
  const next = new Set(set);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}

function blockArgs(option: AlsoBlockOption) {
  return { kind: option.kind, value: option.value };
}
