// import LibraryNotificationDropdown from "./library-notification-dropdown";
import LibraryUploadAgentDialog from "./library-upload-agent-dialog";
import LibrarySearchBar from "./library-search-bar";

interface LibraryActionHeaderProps {}

/**
 * LibraryActionHeader component - Renders a header with search, notifications and filters
 */
const LibraryActionHeader: React.FC<LibraryActionHeaderProps> = ({}) => {
  return (
    <>
      <div className="mb-[32px] hidden items-start justify-between bg-neutral-50 md:flex">
        {/* <LibraryNotificationDropdown /> */}
        <LibrarySearchBar />
        <LibraryUploadAgentDialog />
      </div>

      {/* Mobile and tablet */}
      <div className="flex flex-col gap-4 bg-neutral-50 p-4 pt-[52px] md:hidden">
        <div className="flex w-full justify-between">
          {/* <LibraryNotificationDropdown /> */}
          <LibraryUploadAgentDialog />
        </div>

        <div className="flex items-center justify-center">
          <LibrarySearchBar />
        </div>
      </div>
    </>
  );
};

export default LibraryActionHeader;
