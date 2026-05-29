"use client";

import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { parseAsStringEnum, useQueryState } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";

const sortParser = parseAsStringEnum(Object.values(LibraryAgentSort));

export function useLibraryListPage() {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [librarySortRaw, setLibrarySortRaw] = useQueryState("sort", sortParser);

  // Ensure sort param is always present in URL (even if default)
  useEffect(() => {
    if (!librarySortRaw) {
      setLibrarySortRaw(LibraryAgentSort.updatedAt, { shallow: false });
    }
  }, [librarySortRaw, setLibrarySortRaw]);

  const librarySort = librarySortRaw || LibraryAgentSort.updatedAt;

  const setLibrarySort = useCallback(
    (value: LibraryAgentSort) => {
      setLibrarySortRaw(value, { shallow: false });
    },
    [setLibrarySortRaw],
  );

  return useMemo(
    () => ({
      searchTerm,
      setSearchTerm,
      uploadedFile,
      setUploadedFile,
      librarySort,
      setLibrarySort,
    }),
    [searchTerm, uploadedFile, librarySort, setLibrarySort],
  );
}
