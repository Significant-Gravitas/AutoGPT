"use client";

import FavoritesSection from "./components/FavoritesSection/FavoritesSection";
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
        <FavoritesSection />
        <LibraryAgentList />
      </LibraryPageStateProvider>
    </main>
  );
}
