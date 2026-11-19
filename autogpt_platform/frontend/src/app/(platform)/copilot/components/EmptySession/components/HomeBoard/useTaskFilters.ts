import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { parseAsString, useQueryStates } from "nuqs";
import { useState } from "react";
import { ANY_STATUS, ANYONE, isFiltered } from "./taskFilters";

const SEARCH_DELAY_MS = 300;

const FILTER_PARSERS = {
  q: parseAsString.withDefault(""),
  owner: parseAsString.withDefault(ANYONE),
  status: parseAsString.withDefault(ANY_STATUS),
};

/** The board's search and filters, kept in the address so a reload or a
 *  shared link keeps them. The search box shows exactly what was typed
 *  (its own state, not the address read back), and the search is sent
 *  once typing pauses. */
export function useTaskFilters() {
  const [stored, setStored] = useQueryStates(FILTER_PARSERS, {
    history: "replace",
  });
  const [text, setText] = useState(stored.q);
  const q = useDebouncedValue(text, SEARCH_DELAY_MS);
  const filters = { ...stored, q: text };

  function setQuery(next: string) {
    setText(next);
    void setStored({ q: next || null });
  }

  function reset() {
    setText("");
    void setStored({ q: null, owner: null, status: null });
  }

  return {
    filters,
    /** What the list is read with: the search as last sent. */
    applied: { ...filters, q },
    isFiltered: isFiltered(filters),
    setQuery,
    setOwner: (next: string) =>
      void setStored({ owner: next === ANYONE ? null : next }),
    setStatus: (next: string) =>
      void setStored({ status: next === ANY_STATUS ? null : next }),
    reset,
  };
}
