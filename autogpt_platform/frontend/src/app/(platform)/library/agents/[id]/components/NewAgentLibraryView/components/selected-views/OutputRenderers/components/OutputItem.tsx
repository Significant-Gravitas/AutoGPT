"use client";

import { Text } from "@/components/atoms/Text/Text";
import { OutputMetadata, OutputRenderer } from "../types";

interface OutputItemProps {
  value: any;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
  label?: string;
}

export function OutputItem({
  value,
  metadata,
  renderer,
  label,
}: OutputItemProps) {
  return (
    <div className="relative">
      {label && (
        <Text variant="large-medium" className="capitalize">
          {label}
        </Text>
      )}

      <div className="relative">{renderer.render(value, metadata)}</div>
    </div>
  );
}
