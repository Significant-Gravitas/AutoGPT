"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PlayIcon } from "@hugeicons/core-free-icons";

interface Props {
  isStarting: boolean;
  onStart: () => void;
}

export function StartBar({ isStarting, onStart }: Props) {
  return (
    <div className="flex shrink-0 items-center justify-between gap-3 border-b border-zinc-200 bg-white px-5 py-2.5">
      <Text variant="small" tone="muted">
        This task hasn&apos;t started yet.
      </Text>
      <Button
        size="small"
        leadingIcon={PlayIcon}
        loading={isStarting}
        onClick={onStart}
      >
        Start
      </Button>
    </div>
  );
}
