import { useEffect, useRef } from "react";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { Icon } from "@phosphor-icons/react";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";
import LibraryFolderCreationDialog from "../LibraryFolderCreationDialog/LibraryFolderCreationDialog";

interface LibraryTab {
  id: string;
  title: string;
  icon: Icon;
}

interface Props {
  tabs: LibraryTab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function LibrarySubSection({ tabs, activeTab, onTabChange }: Props) {
  const { registerFavoritesTabRef } = useFavoriteAnimation();
  const favoritesRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    registerFavoritesTabRef(favoritesRef.current);
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
      <LibraryFolderCreationDialog />
    </div>
  );
}
