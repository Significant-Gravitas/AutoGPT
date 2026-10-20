"use client";

import { Text } from "@/components/atoms/Text/Text";
import { EmptySchedulesIllustration } from "./EmptySchedulesIllustration";

export function EmptySchedules() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-20">
      <EmptySchedulesIllustration />
      <div className="flex flex-col items-center gap-2">
        <Text variant="h4" className="text-center text-[1.375rem]">
          Nothing scheduled yet
        </Text>
        <Text variant="large" className="text-zinc-700">
          Create a new run, and you&apos;ll have the option to schedule your
          workflow to run automatically.
        </Text>
      </div>
    </div>
  );
}
