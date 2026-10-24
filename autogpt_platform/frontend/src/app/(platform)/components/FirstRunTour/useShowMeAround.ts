"use client";

import { useRouter } from "next/navigation";
import { useFirstRunTourStore } from "./store";

// "Show me around" in the profile menu: replays the walkthrough from the
// start. It ends on the first tasks, which live on the empty home page, so
// it goes there first from anywhere else (an open chat included).
export function useShowMeAround() {
  const router = useRouter();
  const open = useFirstRunTourStore((state) => state.open);

  return function showMeAround() {
    const { pathname, search } = window.location;
    if (pathname !== "/home" || search !== "") router.push("/home");
    open("menu");
  };
}
