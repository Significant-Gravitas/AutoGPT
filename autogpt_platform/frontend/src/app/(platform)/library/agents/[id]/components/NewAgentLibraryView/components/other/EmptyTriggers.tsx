"use client";

import { Text } from "@/components/atoms/Text/Text";
import { EmptySchedulesIllustration } from "./EmptySchedulesIllustration";

export function EmptyTriggers() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-20">
      <EmptySchedulesIllustration />
      <div className="flex flex-col items-center gap-2">
        <Text variant="h4" className="text-center text-[1.375rem]">
          No triggers yet
        </Text>
        <Text variant="large" className="text-zinc-700">
          Set up automatic triggers for your workflow to run tasks automatically
          — they&apos;ll show up here.
        </Text>
      </div>
    </div>
  );
}
