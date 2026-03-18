import { useEffect, useRef } from "react";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";
import { LibraryTab } from "../../types";
import LibraryFolderCreationDialog from "../LibraryFolderCreationDialog/LibraryFolderCreationDialog";
import { LibrarySortMenu } from "../LibrarySortMenu/LibrarySortMenu";

interface Props {
  tabs: LibraryTab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function LibrarySubSection({
  tabs,
  activeTab,
  onTabChange,
  setLibrarySort,
}: Props) {
  const { registerFavoritesTabRef } = useFavoriteAnimation();
  const favoritesRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    registerFavoritesTabRef(favoritesRef.current);
    return () => {
      registerFavoritesTabRef(null);
    };
  }, [registerFavoritesTabRef]);

  return (
    <div className="flex items-center justify-between gap-4">
      <TabsLine value={activeTab} onValueChange={onTabChange}>
        <TabsLineList>
          {tabs.map((tab) => (
            <TabsLineTrigger
              key={tab.id}
              value={tab.id}
              ref={tab.id === "favorites" ? favoritesRef : undefined}
              className="inline-flex items-center gap-1.5"
            >
              <tab.icon size={16} />
              {tab.title}
            </TabsLineTrigger>
          ))}
        </TabsLineList>
      </TabsLine>
      <LibrarySortMenu setLibrarySort={setLibrarySort} />
      <LibraryFolderCreationDialog />
    </div>
  );
}
