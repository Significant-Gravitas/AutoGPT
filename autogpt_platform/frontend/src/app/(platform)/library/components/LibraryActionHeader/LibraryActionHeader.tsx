import LibraryImportWorkflowDialog from "../LibraryImportWorkflowDialog/LibraryImportWorkflowDialog";
import { LibrarySearchBar } from "../LibrarySearchBar/LibrarySearchBar";
import LibraryUploadAgentDialog from "../LibraryUploadAgentDialog/LibraryUploadAgentDialog";

interface Props {
  setSearchTerm: (value: string) => void;
}

export function LibraryActionHeader({ setSearchTerm }: Props) {
  return (
    <>
      <div className="mb-[32px] hidden items-center justify-center gap-4 md:flex">
        <LibrarySearchBar setSearchTerm={setSearchTerm} />
        <LibraryUploadAgentDialog />
        <LibraryImportWorkflowDialog />
      </div>

      {/* Mobile and tablet */}
      <div className="flex flex-col gap-4 p-4 pt-[52px] md:hidden">
        <div className="flex w-full justify-between gap-2">
          <LibraryUploadAgentDialog />
          <LibraryImportWorkflowDialog />
        </div>

        <div className="flex items-center justify-center">
          <LibrarySearchBar setSearchTerm={setSearchTerm} />
        </div>
      </div>
    </>
  );
}
