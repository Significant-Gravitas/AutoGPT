"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Record } from "@phosphor-icons/react";

interface Props {
  stepCount: number;
}

/**
 * The active-recording indicator shown while a workflow recording is in
 * progress. Distinct from the computer-use indicator: a pulsing red record
 * dot + a live step count, so the user always knows the shim is capturing.
 *
 * Note: the *authoritative* recording indicator is shim-rendered (tray /
 * menu-bar, §9) and the platform can't fake it. This is the in-app
 * companion so the affordance is visible where the user is working.
 */
export function RecordingIndicator({ stepCount }: Props) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="inline-flex items-center gap-1.5 rounded-full border border-red-200 bg-red-50 px-2.5 py-1 text-red-900"
    >
      <Record
        className="h-3.5 w-3.5 animate-pulse text-red-600"
        weight="fill"
      />
      <Text variant="body" className="text-xs font-medium">
        Recording{stepCount > 0 ? ` · ${stepCount} steps` : ""}
      </Text>
    </div>
  );
}
