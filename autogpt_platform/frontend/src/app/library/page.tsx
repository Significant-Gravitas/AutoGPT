import Link from "next/link";

import {
  ArrowBottomRightIcon,
  QuestionMarkCircledIcon,
} from "@radix-ui/react-icons";
import { Alert, AlertDescription } from "@/components/ui/alert";

import { LibraryPageStateProvider } from "./state-provider";
import LibraryActionSubHeader from "@/components/library/library-action-sub-header";
import LibraryActionHeader from "@/components/library/library-action-header";
import LibraryAgentList from "@/components/library/library-agent-list";

/**
 * LibraryPage Component
 * Main component that manages the library interface including agent listing and actions
 */

export default function LibraryPage() {
  return (
    <main className="container min-h-screen space-y-4 sm:px-8 md:px-12">
      <LibraryPageStateProvider>
        {/* Header section containing notifications, search functionality and upload mechanism */}
        <LibraryActionHeader />

        {/* Subheader section containing agent counts and filtering options */}
        <LibraryActionSubHeader />

        {/* Content section displaying agent list with counter and filtering options */}
        <LibraryAgentList />
      </LibraryPageStateProvider>

      <Alert
        variant="default"
        className="fixed bottom-2 left-1/2 hidden max-w-4xl -translate-x-1/2 md:block"
      >
        <AlertDescription className="text-center">
          Prefer the old experience? Click{" "}
          <Link href="/monitoring" className="underline">
            here
          </Link>{" "}
          to go to it. Please do let us know why by clicking the{" "}
          <QuestionMarkCircledIcon className="inline-block size-6 rounded-full bg-[rgba(65,65,64,1)] p-1 align-bottom text-neutral-50" />{" "}
          in the bottom right corner <ArrowBottomRightIcon className="inline" />
        </AlertDescription>
      </Alert>
    </main>
  );
}
