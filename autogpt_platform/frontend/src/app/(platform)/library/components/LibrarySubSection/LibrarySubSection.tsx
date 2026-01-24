import LibraryFolderCreationDialog from "../LibraryFolderCreationDialog/LibraryFolderCreationDialog";
import { LibraryTabs, Tab } from "../LibraryTabs/LibraryTabs";

interface Props {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function LibrarySubSection({ tabs, activeTab, onTabChange }: Props) {
  return (
    <div className="flex justify-between items-center gap-4">
      <LibraryTabs tabs={tabs} activeTab={activeTab} onTabChange={onTabChange} />
      <LibraryFolderCreationDialog />
    </div>
  );
}