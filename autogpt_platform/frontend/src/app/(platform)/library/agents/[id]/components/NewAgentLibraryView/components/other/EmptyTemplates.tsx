"use client";

import { Text } from "@/components/atoms/Text/Text";
import { EmptySchedulesIllustration } from "./EmptySchedulesIllustration";

export function EmptyTemplates() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-20">
      <EmptySchedulesIllustration />
      <div className="flex flex-col items-center gap-2">
        <Text variant="h4" leading="inherit" align="center">
          No templates yet
        </Text>
        <Text variant="large" tone="body">
          Create a task first, then save it as a template to reuse later —
          it&apos;ll show up here.
        </Text>
      </div>
    </div>
  );
}
