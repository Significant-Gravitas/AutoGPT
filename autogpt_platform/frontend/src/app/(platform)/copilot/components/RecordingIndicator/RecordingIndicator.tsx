"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { RecordIcon } from "@hugeicons/core-free-icons";

/**
 * The active-recording indicator shown while a workflow recording is in
 * progress. Distinct from the computer-use indicator: a pulsing red record
 * dot, so the user always knows the shim is capturing.
 *
 * Note: the *authoritative* recording indicator is shim-rendered (tray /
 * menu-bar, §9) and the platform can't fake it. This is the in-app
 * companion so the affordance is visible where the user is working.
 */
export function RecordingIndicator() {
  return (
    <div
      role="status"
      aria-live="polite"
      className="inline-flex items-center gap-1.5 rounded-full border border-error/20 bg-error-surface px-2.5 py-1 text-error-foreground"
    >
      <span className="flex text-error-hover motion-safe:animate-pulse">
        <Icon
          icon={RecordIcon}
          fill="currentColor"
          className="h-3.5 w-3.5"
          aria-hidden="true"
        />
      </span>
      <Text variant="small-medium-tight">Recording</Text>
    </div>
  );
}
