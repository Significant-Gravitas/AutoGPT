import { useListCollectionRows } from "@/app/api/__generated__/endpoints/collections/collections";
import type { Condition } from "@/app/api/__generated__/models/condition";
import type { Sort } from "@/app/api/__generated__/models/sort";
import { okData } from "@/app/api/helpers";
import { keepPreviousData } from "@tanstack/react-query";
import { useState } from "react";
import {
  chipColumn,
  type TableColumn,
  type TableRow,
  whereParam,
} from "./helpers";
import { useRowWrites } from "./useRowWrites";

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
  const writes = useRowWrites(collectionID);
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

  function toggleChip(option: string) {
    setChips((current) =>
      current.includes(option)
        ? current.filter((o) => o !== option)
        : [...current, option],
    );
  }

  return {
    ...writes,
    search,
    setSearch,
    chips,
    toggleChip,
    chipField,
    rows: narrowed ? (narrowedRows.data?.rows ?? []) : rows,
    total: narrowed ? (narrowedRows.data?.total ?? 0) : total,
    isNarrowing: narrowed && narrowedRows.isFetching,
  };
}
