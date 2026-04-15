import LibraryImportDialog from "../LibraryImportDialog/LibraryImportDialog";
import { LibrarySearchBar } from "../LibrarySearchBar/LibrarySearchBar";

interface Props {
  setSearchTerm: (value: string) => void;
}

export function LibraryActionHeader({ setSearchTerm }: Props) {
  return (
    <>
      <div className="mb-[32px] hidden items-center justify-center gap-4 md:flex">
        <LibrarySearchBar setSearchTerm={setSearchTerm} />
        <LibraryImportDialog />
      </div>

      {/* Mobile and tablet */}
      <div className="flex flex-col gap-4 p-4 pt-[52px] md:hidden">
        <div className="flex w-full justify-between gap-2">
          <LibraryImportDialog />
        </div>

        <div className="flex items-center justify-center">
          <LibrarySearchBar setSearchTerm={setSearchTerm} />
        </div>
      </div>
    </>
  );
}
