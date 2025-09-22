import { debounce } from "lodash";
import { useCallback, useEffect, useRef, useState } from "react";
import { useBlockMenuContext } from "../block-menu-provider";

const SEARCH_DEBOUNCE_MS = 300;

export const useBlockMenuSearchBar = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [localQuery, setLocalQuery] = useState("");
  const { setSearchQuery, setSearchId, searchId } = useBlockMenuContext();

  const searchIdRef = useRef(searchId);
  useEffect(() => {
    searchIdRef.current = searchId;
  }, [searchId]);

  const debouncedSetSearchQuery = useCallback(
    debounce((value: string) => {
      setSearchQuery(value);
      if (value.length === 0) {
        setSearchId(undefined);
      } else if (!searchIdRef.current) {
        setSearchId(crypto.randomUUID());
      }
    }, SEARCH_DEBOUNCE_MS),
    [setSearchQuery, setSearchId],
  );

  useEffect(() => {
    return () => {
      debouncedSetSearchQuery.cancel();
    };
  }, [debouncedSetSearchQuery]);

  const handleClear = () => {
    setLocalQuery("");
    setSearchQuery("");
    setSearchId(undefined);
    debouncedSetSearchQuery.cancel();
  };

  return {
    handleClear,
    inputRef,
    localQuery,
    setLocalQuery,
    debouncedSetSearchQuery,
  };
};
