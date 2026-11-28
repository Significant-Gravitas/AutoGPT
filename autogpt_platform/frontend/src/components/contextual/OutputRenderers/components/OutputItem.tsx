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
        <Text
          variant="large"
          weight="medium"
          transform="capitalize"
          data-testid="output-item-label"
        >
          {label}
        </Text>
      )}

      <div className="relative" data-testid="output-item-value">
        {renderer.render(value, metadata)}
      </div>
    </div>
  );
}
