import { LibraryNotificationDropdown } from "../LibraryNotificationDropdown";

const LibraryActionHeader: React.FC = () => {
  return (
    <div className="flex w-screen items-center justify-between px-4 pt-6">
      <LibraryNotificationDropdown />
      <LibrarySearchBar />
      <LibraryUploadAgent />
    </div>
  );
};

const LibrarySearchBar = () => {
  return (
    <div>
      SearchBar
      {/* Search bar content */}
    </div>
  );
};

const LibraryUploadAgent = () => {
  return <div>Uploading Agent</div>;
};

export default LibraryActionHeader;
