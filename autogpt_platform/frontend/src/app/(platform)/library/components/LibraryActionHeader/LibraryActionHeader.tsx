import LibrarySearchBar from "../LibrarySearchBar/LibrarySearchBar";
import LibraryUploadAgentDialog from "../LibraryUploadAgentDialog/LibraryUploadAgentDialog";

export function LibraryActionHeader() {
  return (
    <>
      <div className="mb-[32px] hidden items-center justify-center gap-4 md:flex">
        <LibrarySearchBar />
        <LibraryUploadAgentDialog />
      </div>

      {/* Mobile and tablet */}
      <div className="flex flex-col gap-4 p-4 pt-[52px] md:hidden">
        <div className="flex w-full justify-between">
          <LibraryUploadAgentDialog />
        </div>

        <div className="flex items-center justify-center">
          <LibrarySearchBar />
        </div>
      </div>
    </>
  );
}
