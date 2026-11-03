"use client";

import { useToast } from "@/components/molecules/Toast/use-toast";
import { authClient } from "@/lib/auth/client";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  type AdminUser,
  type BanDuration,
  getBanDurationSeconds,
  parseLookupQuery,
  toAdminUser,
} from "./helpers";
import { getAdminUserLookupQueryKey } from "./useAdminUserLookup";

interface BanArgs {
  reason: string;
  duration: BanDuration;
}

export function useUserBan(user: AdminUser) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  function storeUpdatedUser(updated: AdminUser) {
    for (const lookup of [updated.id, updated.email]) {
      queryClient.setQueryData(
        getAdminUserLookupQueryKey(parseLookupQuery(lookup)),
        updated,
      );
    }
  }

  const ban = useMutation({
    mutationFn: async ({ reason, duration }: BanArgs) => {
      const banReason = reason.trim();
      if (!banReason) throw new Error("Give a reason for the ban.");
      const { data, error } = await authClient.admin.banUser({
        userId: user.id,
        banReason,
        banExpiresIn: getBanDurationSeconds(duration),
      });
      if (error || !data) {
        throw new Error(error?.message || "Failed to ban the user");
      }
      return toAdminUser(data.user);
    },
    onSuccess: (updated) => {
      storeUpdatedUser(updated);
      toast({
        title: `Banned ${updated.email}`,
        description:
          "Their sessions are signed out. Tokens they hold or get in the next 5 minutes work for up to an hour, and their API keys, schedules and triggers keep running.",
        variant: "success",
      });
    },
    onError: (error) => {
      toast({
        title: "Couldn't ban the user",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const unban = useMutation({
    mutationFn: async () => {
      const { data, error } = await authClient.admin.unbanUser({
        userId: user.id,
      });
      if (error || !data) {
        throw new Error(error?.message || "Failed to unban the user");
      }
      return toAdminUser(data.user);
    },
    onSuccess: (updated) => {
      storeUpdatedUser(updated);
      toast({ title: `Unbanned ${updated.email}`, variant: "success" });
    },
    onError: (error) => {
      toast({
        title: "Couldn't unban the user",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  return {
    banUser: ban.mutate,
    unbanUser: unban.mutate,
    isBanning: ban.isPending,
    isUnbanning: unban.isPending,
  };
}
