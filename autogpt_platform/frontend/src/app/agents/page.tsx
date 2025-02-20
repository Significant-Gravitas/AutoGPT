import { LibraryActionSubHeader } from "@/components/agptui/composite/LibraryActionSubHeader";
import LibraryAgentListContainer from "@/components/agptui/composite/LibraryAgentListContainer";
import LibraryActionHeader from "@/components/agptui/composite/LibraryActionHeader";

/**
 * LibraryPage Component
 * Main component that manages the library interface including agent listing and actions
 */

export default function LibraryPage() {
  return (
    <main className="mx-auto w-screen max-w-[1600px] space-y-4 bg-neutral-50 p-4 px-2 dark:bg-neutral-900 sm:px-8 md:px-12">
      {/* Header section containing notifications, search functionality and upload mechanism */}
      <LibraryActionHeader />

      {/* Subheader section containing agent counts and filtering options */}
      <LibraryActionSubHeader />

      {/* Content section displaying agent list with counter and filtering options */}
      <LibraryAgentListContainer />
    </main>
  );
}
