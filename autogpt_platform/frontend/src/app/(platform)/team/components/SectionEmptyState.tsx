import type { ComponentProps } from "react";
import { PanelEmptyState } from "@/app/(platform)/copilot/components/ContextPanel/components/PanelEmptyState/PanelEmptyState";

/** The chat side panel's tilted-tiles empty state, set on a card so it sits
 *  under a profile tab's toolbar the way a populated list would. */
export function SectionEmptyState(
  props: ComponentProps<typeof PanelEmptyState>,
) {
  return (
    <div className="mt-4 rounded-2xl bg-surface smooth-shadow-ring-sm">
      <PanelEmptyState textSize="body-md" {...props} />
    </div>
  );
}
