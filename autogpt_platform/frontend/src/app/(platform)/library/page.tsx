"use client";

import { useEffect, useState, useCallback } from "react";
import { HeartIcon, ListIcon } from "@phosphor-icons/react";
import { LibraryActionHeader } from "./components/LibraryActionHeader/LibraryActionHeader";
import { LibraryAgentList } from "./components/LibraryAgentList/LibraryAgentList";
import { Tab } from "./components/LibraryTabs/LibraryTabs";
import { useLibraryListPage } from "./components/useLibraryListPage";
import { FavoriteAnimationProvider } from "./context/FavoriteAnimationContext";

const LIBRARY_TABS: Tab[] = [
  { id: "all", title: "All", icon: ListIcon },
  { id: "favorites", title: "Favorites", icon: HeartIcon },
];

export default function LibraryPage() {
  const { searchTerm, setSearchTerm, librarySort, setLibrarySort } =
    useLibraryListPage();
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(LIBRARY_TABS[0].id);

  useEffect(() => {
    document.title = "Library – AutoGPT Platform";
  }, []);

  function handleTabChange(tabId: string) {
    setActiveTab(tabId);
    setSelectedFolderId(null);
  }

  const handleFavoriteAnimationComplete = useCallback(() => {
    setActiveTab("favorites");
    setSelectedFolderId(null);
  }, []);

  return (
    <FavoriteAnimationProvider
      onAnimationComplete={handleFavoriteAnimationComplete}
    >
      <main className="pt-160 container min-h-screen space-y-4 pb-20 pt-16 sm:px-8 md:px-12">
        <LibraryActionHeader setSearchTerm={setSearchTerm} />
        <LibraryAgentList
          searchTerm={searchTerm}
          librarySort={librarySort}
          setLibrarySort={setLibrarySort}
          selectedFolderId={selectedFolderId}
          onFolderSelect={setSelectedFolderId}
          tabs={LIBRARY_TABS}
          activeTab={activeTab}
          onTabChange={handleTabChange}
        />
      </main>
    </FavoriteAnimationProvider>
  );
}
