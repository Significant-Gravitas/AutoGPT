import { track } from "@/lib/analytics/client";
import { EVENTS } from "@/lib/analytics/rules";
import { useMountEffect } from "@/lib/useMountEffect";

/** Mounted once Home has its data, so the view counts once per visit that showed it. */
export function HomeViewed() {
  useMountEffect(() => {
    track(EVENTS.homeViewed);
  });
  return null;
}
