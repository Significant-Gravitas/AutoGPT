"use client";

import { useEffect } from "react";
import { FavoritesSection } from "./components/FavoritesSection/FavoritesSection";
import { LibraryActionHeader } from "./components/LibraryActionHeader/LibraryActionHeader";
import { LibraryAgentList } from "./components/LibraryAgentList/LibraryAgentList";
import { useLibraryListPage } from "./components/useLibraryListPage";

export default function LibraryPage() {
  const { searchTerm, setSearchTerm, librarySort, setLibrarySort } =
    useLibraryListPage();

  useEffect(() => {
    document.title = "Library – AutoGPT Platform";
  }, []);

  return (
    <main className="pt-160 container min-h-screen space-y-4 pb-20 pt-16 sm:px-8 md:px-12">
      <LibraryActionHeader setSearchTerm={setSearchTerm} />
      <FavoritesSection searchTerm={searchTerm} />
      <LibraryAgentList
        searchTerm={searchTerm}
        librarySort={librarySort}
        setLibrarySort={setLibrarySort}
      />
    </main>
  );
}
