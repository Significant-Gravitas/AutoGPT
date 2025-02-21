import Link from "next/link";

import {
  ArrowBottomRightIcon,
  QuestionMarkCircledIcon,
} from "@radix-ui/react-icons";

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

      <p className="w-full p-4 text-center text-gray-500">
        Prefer the old experience? Click{" "}
        <Link href="/monitoring" className="underline">
          here
        </Link>{" "}
        to go to it. Please do let us know why by clicking the{" "}
        <QuestionMarkCircledIcon className="inline-block size-6 rounded-full bg-[rgba(65,65,64,1)] p-1 text-neutral-50" />{" "}
        in the bottom right corner <ArrowBottomRightIcon className="inline" />
      </p>
    </main>
  );
}
