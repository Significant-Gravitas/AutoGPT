import { useGetV1ListAirtableBases } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { okData } from "@/app/api/helpers";
import { useState } from "react";
import { loadErrorText, matchesSearch } from "./helpers";

export function useAirtableBasePickerField(credentialsID: string) {
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const query = useGetV1ListAirtableBases(credentialsID, {
    query: { select: okData, retry: false },
  });
  const bases = query.data?.bases ?? [];

  // A search from the last time the list was open shouldn't hide bases now.
  function changeOpen(next: boolean) {
    setOpen(next);
    if (!next) setSearch("");
  }

  return {
    bases,
    shown: bases.filter((base) => matchesSearch(base.name, search)),
    truncated: query.data?.truncated ?? false,
    isLoading: query.isLoading,
    isRefreshing: query.isFetching && !query.isLoading,
    error: query.isError ? loadErrorText(query.error) : null,
    search,
    setSearch,
    open,
    setOpen: changeOpen,
    refresh: () => query.refetch(),
  };
}
