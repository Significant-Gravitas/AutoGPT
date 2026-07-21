import { useEffect, useRef, useState } from "react";
import { filterSessions, type SearchSession } from "./helpers";

const DEBOUNCE_MS = 150;

export function useChatSearch(sessions: SearchSession[], isOpen: boolean) {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const highlightedResultRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setQuery("");
      setDebouncedQuery("");
      setHighlightedIndex(0);
    }
  }, [isOpen]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setDebouncedQuery(query);
    }, DEBOUNCE_MS);

    return () => window.clearTimeout(timeout);
  }, [query]);

  const results = filterSessions(sessions, debouncedQuery);

  useEffect(() => {
    setHighlightedIndex(0);
  }, [debouncedQuery, sessions.length]);

  useEffect(() => {
    if (highlightedIndex >= results.length) {
      setHighlightedIndex(Math.max(0, results.length - 1));
    }
  }, [highlightedIndex, results.length]);

  useEffect(() => {
    highlightedResultRef.current?.scrollIntoView({
      block: "nearest",
    });
  }, [highlightedIndex]);

  function moveHighlight(direction: 1 | -1) {
    if (results.length === 0) return;
    setHighlightedIndex((current) => {
      const next = current + direction;
      if (next < 0 || next >= results.length) return current;
      return next;
    });
  }

  function clearQuery() {
    setQuery("");
    setDebouncedQuery("");
  }

  const isSearching = debouncedQuery.trim().length > 0;

  return {
    query,
    debouncedQuery,
    setQuery,
    clearQuery,
    results,
    highlightedIndex,
    setHighlightedIndex,
    highlightedResultRef,
    moveHighlight,
    isSearching,
  };
}
