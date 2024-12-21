import LibraryActionHeader from "@/components/agptui/composite/LibraryActionHeader";

const LibraryPage = () => {
  return (
    <main className="p-4">
      {/* Top section - includes notification, search and uploading mechansim */}
      <LibraryActionHeader />

      {/* Last section for Agent Lists, Agent counter and filter */}
      <div></div>
    </main>
  );
};

export default LibraryPage;
