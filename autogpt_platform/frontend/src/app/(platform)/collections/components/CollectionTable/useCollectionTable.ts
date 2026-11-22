import {
  useAcceptCollectionRowValues,
  useAddCollectionRow,
  useDeleteCollectionRow,
  useEditCollectionRow,
  useListCollectionRows,
} from "@/app/api/__generated__/endpoints/collections/collections";
import type { Condition } from "@/app/api/__generated__/models/condition";
import type { Sort } from "@/app/api/__generated__/models/sort";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { keepPreviousData, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  chipColumn,
  type TableColumn,
  type TableRow,
  whereParam,
} from "./helpers";

interface Args {
  collectionID: string;
  columns: TableColumn[];
  rows: TableRow[];
  total: number;
  filter: Condition[];
  sort: Sort | null;
}

export function useCollectionTable({
  collectionID,
  columns,
  rows,
  total,
  filter,
  sort,
}: Args) {
  const [search, setSearch] = useState("");
  const [chips, setChips] = useState<string[]>([]);
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const chipField = chipColumn(columns);
  const narrowed = Boolean(search.trim()) || chips.length > 0;

  const narrowedRows = useListCollectionRows(
    collectionID,
    {
      search: search.trim() || undefined,
      where: whereParam(filter, chipField?.field ?? null, chips),
      sort: sort?.field,
      direction: sort?.direction ?? "asc",
      limit: 50,
    },
    {
      query: {
        enabled: narrowed,
        select: okData,
        placeholderData: keepPreviousData,
      },
    },
  );

  function refresh() {
    return queryClient.invalidateQueries({
      predicate: (query) => {
        const key = query.queryKey[0];
        return (
          typeof key === "string" &&
          (key.startsWith("/api/pages") || key.startsWith("/api/collections"))
        );
      },
    });
  }

  function onWriteError(error: unknown) {
    if (error instanceof ApiError && error.status === 409) {
      toast({
        title: "This row changed since you opened it",
        description: "It shows the new values now.",
      });
    } else {
      toast({
        title: "Couldn't save that",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
    }
    refresh();
  }

  const mutation = { onSuccess: refresh, onError: onWriteError };
  const edit = useEditCollectionRow({ mutation });
  const add = useAddCollectionRow({ mutation });
  const remove = useDeleteCollectionRow({ mutation });
  const accept = useAcceptCollectionRowValues({ mutation });

  function toggleChip(option: string) {
    setChips((current) =>
      current.includes(option)
        ? current.filter((o) => o !== option)
        : [...current, option],
    );
  }

  return {
    search,
    setSearch,
    chips,
    toggleChip,
    chipField,
    rows: narrowed ? (narrowedRows.data?.rows ?? []) : rows,
    total: narrowed ? (narrowedRows.data?.total ?? 0) : total,
    isNarrowing: narrowed && narrowedRows.isFetching,
    isSaving:
      edit.isPending || add.isPending || remove.isPending || accept.isPending,
    saveCell(row: TableRow, field: string, value: unknown) {
      return settled(
        edit.mutateAsync({
          collectionId: collectionID,
          rowId: row.id,
          data: { values: { [field]: value }, based_on: row.revision },
        }),
      );
    },
    addRow(values: Record<string, unknown>) {
      return settled(
        add.mutateAsync({ collectionId: collectionID, data: { values } }),
      );
    },
    deleteRow(row: TableRow) {
      return settled(
        remove.mutateAsync({ collectionId: collectionID, rowId: row.id }),
      );
    },
    acceptPending(row: TableRow, field: string) {
      return settled(
        accept.mutateAsync({
          collectionId: collectionID,
          rowId: row.id,
          data: { fields: [field], based_on: row.revision },
        }),
      );
    },
  };
}

/** Whether the write landed; a refusal has already been shown as a toast. */
async function settled(write: Promise<unknown>) {
  try {
    await write;
    return true;
  } catch {
    return false;
  }
}
