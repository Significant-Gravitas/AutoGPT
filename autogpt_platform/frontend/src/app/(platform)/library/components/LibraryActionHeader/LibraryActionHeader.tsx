import LibraryImportDialog from "../LibraryImportDialog/LibraryImportDialog";
import { LibrarySearchBar } from "../LibrarySearchBar/LibrarySearchBar";

interface Props {
  setSearchTerm: (value: string) => void;
  isImportOpen: boolean;
  onImportOpenChange: (open: boolean) => void;
}

export function LibraryActionHeader({
  setSearchTerm,
  isImportOpen,
  onImportOpenChange,
}: Props) {
  return (
    <div className="flex flex-col gap-4 p-4 pt-[52px] md:mb-7 md:flex-row md:items-center md:justify-center md:p-0">
      <div className="order-2 flex items-center justify-center md:order-1">
        <LibrarySearchBar setSearchTerm={setSearchTerm} />
      </div>
      <div className="order-1 flex items-center justify-between md:order-2">
        <LibraryImportDialog
          isOpen={isImportOpen}
          onOpenChange={onImportOpenChange}
        />
      </div>
    </div>
  );
}
