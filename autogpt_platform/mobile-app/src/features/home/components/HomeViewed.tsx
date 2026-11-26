import { track } from "@/lib/analytics/client";
import { EVENTS } from "@/lib/analytics/rules";
import { useMountEffect } from "@/lib/useMountEffect";

/** Mounted once Home has its data; Home stays mounted, so this fires once per launch. */
export function HomeViewed() {
  useMountEffect(() => {
    track(EVENTS.homeViewed);
  });
  return null;
}
