"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type BanDuration, getBanDurationSeconds } from "./helpers";
import { isIPReportQuery } from "./useAccessBlocks";

export interface BanAccountResult {
  userId: string;
  ok: boolean;
  error?: string;
}

interface BanArgs {
  userIds: string[];
  reason: string;
  duration: BanDuration;
}

// One request for the whole list (api/admin/ban-accounts), answered per
// account, so a failure part-way through says exactly who wasn't banned.
export function useBanAccounts() {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: async ({ userIds, reason, duration }: BanArgs) => {
      const response = await fetch("/api/admin/ban-accounts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userIds,
          reason: reason.trim(),
          banExpiresIn: getBanDurationSeconds(duration),
        }),
      });
      const body = (await response.json().catch(() => null)) as {
        results?: BanAccountResult[];
        error?: string;
      } | null;
      if (!response.ok || !body?.results) {
        throw new Error(body?.error || "Failed to ban the accounts");
      }
      return body.results;
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ predicate: isIPReportQuery });
      void queryClient.invalidateQueries({
        queryKey: ["admin", "user-lookup"],
      });
    },
  });

  return { banAccounts: mutation.mutateAsync, isBanning: mutation.isPending };
}
