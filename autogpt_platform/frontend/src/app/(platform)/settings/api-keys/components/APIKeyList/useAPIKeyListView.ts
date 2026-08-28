"use client";

import { useMemo, useState } from "react";

import { useAPIKeysList } from "../hooks/useAPIKeysList";
import { useAPIKeySelection } from "./useAPIKeySelection";
import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";

export function useAPIKeyListView() {
  const list = useAPIKeysList();
  // Stabilise the id array so useAPIKeySelection's effect doesn't re-run on
  // every parent render (the effect only cares when the set of ids changes).
  const allIds = useMemo(() => list.keys.map((key) => key.id), [list.keys]);
  const selection = useAPIKeySelection(allIds);
  const [deleteTarget, setDeleteTarget] = useState<APIKeyInfo[] | null>(null);

  function requestDelete(ids: string[]) {
    const idSet = new Set(ids);
    setDeleteTarget(list.keys.filter((key) => idSet.has(key.id)));
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
    selection,
    deleteTarget,
    requestDelete,
    closeDeleteDialog,
    handleDeleted,
  };
}
