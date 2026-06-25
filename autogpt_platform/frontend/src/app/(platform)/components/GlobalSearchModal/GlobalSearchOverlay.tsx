"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { GlobalSearchModal } from "./GlobalSearchModal";
import { selectSearchResult } from "./selectSearchResult";
import { useGlobalSearchStore } from "./useGlobalSearchStore";

// Mounted once in the platform layout so Cmd/Ctrl+K opens the search palette
// from any page. Renders a single modal instance driven by the shared store.
export function GlobalSearchOverlay() {
  const isEnabled = useGetFlag(Flag.CHAT_SEARCH);
  const router = useRouter();
  const pathname = usePathname();
  const isOpen = useGlobalSearchStore((state) => state.isOpen);
  const closeSearch = useGlobalSearchStore((state) => state.closeSearch);

  // Close the palette once navigation lands. The modal now lives in the
  // platform layout (persists across pages), so route changes no longer
  // unmount it — close on a real pathname change instead.
  const previousPathname = useRef(pathname);
  useEffect(() => {
    if (previousPathname.current !== pathname) {
      previousPathname.current = pathname;
      useGlobalSearchStore.getState().closeSearch();
    }
  }, [pathname]);

  useEffect(() => {
    if (!isEnabled) return;
    function handleSearchShortcut(event: KeyboardEvent) {
      if (event.repeat) return;
      if (event.key.toLocaleLowerCase() !== "k") return;
      if (!event.metaKey && !event.ctrlKey) return;
      event.preventDefault();
      useGlobalSearchStore.getState().toggleSearch();
    }

    document.addEventListener("keydown", handleSearchShortcut);
    return () => document.removeEventListener("keydown", handleSearchShortcut);
  }, [isEnabled]);

  if (!isEnabled) return null;

  return (
    <GlobalSearchModal
      isOpen={isOpen}
      onClose={closeSearch}
      onSelectItem={(item) => {
        // Close on selection directly — a chat_session selected while already
        // on /copilot only changes the query param, so the pathname-change
        // effect above wouldn't fire.
        closeSearch();
        selectSearchResult(router, item);
      }}
    />
  );
}
