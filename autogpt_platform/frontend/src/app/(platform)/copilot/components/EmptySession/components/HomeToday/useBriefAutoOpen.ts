import { useHomePage } from "@/app/(platform)/home/useHomePage";
import { Key, storage } from "@/services/storage/local-storage";
import { useEffect, useState } from "react";
import { useIsMobile } from "../../../../useIsMobile";
import { useWorkAreaStore } from "../../../../workArea";
import { briefName } from "./helpers";
import { useTaskSuggestions } from "./useTaskSuggestions";

/** On the first visit to Home each local day, opens the day's brief beside
 *  it, once: the day is remembered, so a brief closed that day stays closed.
 *  Only when there is something to read, and never on a phone, which has no
 *  sidebar. */
export function useBriefAutoOpen(enabled: boolean) {
  const [now] = useState(() => Date.now());
  const day = new Date(now).toDateString();
  const isMobile = useIsMobile();
  const openBrief = useWorkAreaStore((s) => s.openBrief);
  const { dashboard } = useHomePage();
  const { suggestions } = useTaskSuggestions();
  const hasNews = Boolean(dashboard?.briefing) || suggestions.length > 0;

  useEffect(() => {
    if (!enabled || isMobile || !hasNews) return;
    if (storage.get(Key.HOME_BRIEF_AUTO_OPENED) === day) return;
    storage.set(Key.HOME_BRIEF_AUTO_OPENED, day);
    storage.set(Key.HOME_BRIEF_SEEN, day);
    openBrief(briefName(now));
  }, [enabled, isMobile, hasNews, day, now, openBrief]);
}
