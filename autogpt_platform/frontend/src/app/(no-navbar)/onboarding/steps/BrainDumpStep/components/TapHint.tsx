"use client";

import { Text } from "@/components/atoms/Text/Text";

export function TapHint({ caption }: { caption: string }) {
  return (
    <Text variant="lead" className="text-center !text-base !text-zinc-500">
      {caption}
    </Text>
  );
}
