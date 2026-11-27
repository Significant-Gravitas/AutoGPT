import { track } from "@/lib/analytics/client";
import { EVENTS } from "@/lib/analytics/rules";
import { useMountEffect } from "@/lib/useMountEffect";

/** Sends a view when the signed-in Home screen mounts with data. */
export function HomeViewed() {
  useMountEffect(() => {
    track(EVENTS.homeViewed);
  });
  return null;
}
