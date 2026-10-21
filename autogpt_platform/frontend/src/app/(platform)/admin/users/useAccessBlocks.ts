"use client";

import {
  getGetAdminListAccessBlocksQueryKey,
  useDeleteAdminDeleteAccessBlock,
  usePostAdminCreateAccessBlock,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { AuthAccessBlockKind } from "@/app/api/__generated__/models/authAccessBlockKind";
import { type Query, useQueryClient } from "@tanstack/react-query";

export interface CreateBlockArgs {
  kind: AuthAccessBlockKind;
  value: string;
  reason: string;
  expiresAt: Date | null;
}

export function useAccessBlocks() {
  const queryClient = useQueryClient();
  const create = usePostAdminCreateAccessBlock();
  const remove = useDeleteAdminDeleteAccessBlock();

  function refreshBlocks() {
    void queryClient.invalidateQueries({
      queryKey: getGetAdminListAccessBlocksQueryKey().slice(0, 1),
    });
    void queryClient.invalidateQueries({ predicate: isIPReportQuery });
  }

  async function createBlock({
    kind,
    value,
    reason,
    expiresAt,
  }: CreateBlockArgs) {
    try {
      return await create.mutateAsync({
        data: {
          kind,
          value,
          reason: reason.trim(),
          expires_at: expiresAt,
        },
      });
    } finally {
      refreshBlocks();
    }
  }

  async function deleteBlock(blockId: string) {
    try {
      return await remove.mutateAsync({ blockId });
    } finally {
      refreshBlocks();
    }
  }

  return {
    createBlock,
    deleteBlock,
    isCreating: create.isPending,
    isDeleting: remove.isPending,
  };
}

// Every account's report: a block or a ban changes the badges on all of them.
export function isIPReportQuery(query: Query) {
  const [key] = query.queryKey;
  return (
    typeof key === "string" &&
    key.startsWith("/api/admin/users/") &&
    key.endsWith("/ip-report")
  );
}
