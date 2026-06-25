"use client";

import { Text } from "@/components/atoms/Text/Text";
import { CubeIcon } from "@phosphor-icons/react";
import { StepStatusIcon } from "../helpers";

interface Props {
  index: number;
  description: string;
  blockName?: string | null;
  status: string;
}

export function StepItem({ index, description, blockName, status }: Props) {
  return (
    <div className="flex items-start gap-3 py-1.5">
      <div className="mt-0.5 flex shrink-0 items-center">
        <StepStatusIcon status={status} />
      </div>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="text-sm text-foreground">
          {index + 1}. {description}
        </Text>
        {blockName && (
          <div className="mt-0.5 flex items-center gap-1">
            <CubeIcon size={12} className="text-muted-foreground" />
            <Text
              variant="small"
              className="font-mono text-xs text-muted-foreground"
            >
              {blockName}
            </Text>
          </div>
        )}
      </div>
    </div>
  );
}
