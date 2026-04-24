"use client";

import { useState } from "react";

import { useAPIKeysList } from "../hooks/useAPIKeysList";
import { useAPIKeySelection } from "./useAPIKeySelection";

export function useAPIKeyListView() {
  const list = useAPIKeysList();
  const allIds = list.keys.map((key) => key.id);
  const selection = useAPIKeySelection(allIds);
  const [deleteTarget, setDeleteTarget] = useState<string[] | null>(null);

  function requestDelete(ids: string[]) {
    setDeleteTarget(ids);
  }

  function closeDeleteDialog(open: boolean) {
    if (!open) setDeleteTarget(null);
  }

  function handleDeleted() {
    selection.clear();
  }

  return {
    keys: list.keys,
    isLoading: list.isLoading,
    isError: list.isError,
    error: list.error,
    refetch: list.refetch,
    isEmpty: list.isEmpty,
    hasNextPage: list.hasNextPage,
    isFetchingNextPage: list.isFetchingNextPage,
    fetchNextPage: list.fetchNextPage,
    selection,
    deleteTarget,
    requestDelete,
    closeDeleteDialog,
    handleDeleted,
  };
}
