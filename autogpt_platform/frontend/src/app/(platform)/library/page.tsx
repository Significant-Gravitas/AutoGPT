"use client";
import Link from "next/link";

import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  ArrowBottomRightIcon,
  QuestionMarkCircledIcon,
} from "@radix-ui/react-icons";

import LibraryActionHeader from "./components/LibraryActionHeader/LibraryActionHeader";
import LibraryAgentList from "./components/LibraryAgentList/LibraryAgentList";
import { LibraryPageStateProvider } from "./components/state-provider";

/**
 * LibraryPage Component
 * Main component that manages the library interface including agent listing and actions
 */
export default function LibraryPage() {
  return (
    <main className="pt-160 container min-h-screen space-y-4 pb-20 pt-16 sm:px-8 md:px-12">
      <LibraryPageStateProvider>
        <LibraryActionHeader />
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
