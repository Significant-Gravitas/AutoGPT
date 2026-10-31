"use client";

import { Text } from "@/components/atoms/Text/Text";
import { EmptySchedulesIllustration } from "./EmptySchedulesIllustration";

export function EmptySchedules() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-20">
      <EmptySchedulesIllustration />
      <div className="flex flex-col items-center gap-2">
        <Text
          variant="h4"
          // oxlint-disable-next-line design-system/no-restyle -- no heading variant keeps the parent line height; h4 alone would move the empty state by 9px
          className="text-center leading-inherit"
        >
          Nothing scheduled yet
        </Text>
        <Text variant="large" tone="body">
          Create a new run, and you&apos;ll have the option to schedule your
          workflow to run automatically.
        </Text>
      </div>
    </div>
  );
}
