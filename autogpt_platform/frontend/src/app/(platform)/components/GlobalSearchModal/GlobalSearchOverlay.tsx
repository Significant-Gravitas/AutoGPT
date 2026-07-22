"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { GlobalSearchModal } from "./GlobalSearchModal";
import { selectSearchResult } from "./selectSearchResult";
import { useGlobalSearchStore } from "./useGlobalSearchStore";

// Mounted once in the platform layout so Cmd/Ctrl+Shift+K opens the search
// palette from any page. Renders a single modal instance driven by the store.
export function GlobalSearchOverlay() {
  // Search is a first-class part of the new layout (sidebar Search item +
  // Cmd/Ctrl+Shift+K), so enable it whenever that layout is on, independent of
  // chat-search flag that gates it in the classic layout.
  const isChatSearchEnabled = useGetFlag(Flag.CHAT_SEARCH);
  const isNewLayoutEnabled = useGetFlag(Flag.AUTOGPT_NEW_LAYOUT);
  const isEnabled = isChatSearchEnabled || isNewLayoutEnabled;
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
      // The new layout binds Cmd/Ctrl+Shift+K; the classic chat-search path
      // keeps its original plain Cmd/Ctrl+K, so only require Shift on the new
      // layout to avoid changing the classic binding.
      if (isNewLayoutEnabled && !event.shiftKey) return;
      event.preventDefault();
      useGlobalSearchStore.getState().toggleSearch();
    }

    document.addEventListener("keydown", handleSearchShortcut);
    return () => document.removeEventListener("keydown", handleSearchShortcut);
  }, [isEnabled, isNewLayoutEnabled]);

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
