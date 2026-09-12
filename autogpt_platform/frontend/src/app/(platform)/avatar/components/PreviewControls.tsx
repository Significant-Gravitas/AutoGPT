"use client";

import { Text } from "@/components/atoms/Text/Text";
import type { AvatarStatus } from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";
import {
  EXPRESSION_OPTIONS,
  PREVIEW_SIZES,
  STATUS_OPTIONS,
  type ExpressionChoice,
  type PreviewSize,
} from "../helpers";

interface Props {
  status: AvatarStatus;
  onStatusChange: (status: AvatarStatus) => void;
  expression: ExpressionChoice;
  onExpressionChange: (expression: ExpressionChoice) => void;
  previewSize: PreviewSize;
  onPreviewSizeChange: (size: PreviewSize) => void;
}

const CHIP =
  "rounded-full border px-3 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2";

export function PreviewControls({
  status,
  onStatusChange,
  expression,
  onExpressionChange,
  previewSize,
  onPreviewSizeChange,
}: Props) {
  return (
    <section className="flex flex-col gap-4 rounded-2xl border border-zinc-200 bg-white p-4">
      <div className="flex flex-col gap-2">
        <Text variant="small" tone="muted" as="h2">
          Status
        </Text>
        <div className="flex flex-wrap gap-2">
          {STATUS_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              aria-pressed={option.id === status}
              onClick={() => onStatusChange(option.id)}
              className={cn(
                CHIP,
                option.id === status
                  ? "border-zinc-900 bg-zinc-900 text-white"
                  : "border-zinc-200 text-zinc-700 hover:border-zinc-400",
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <Text variant="small" tone="muted" as="h2">
          Expression
        </Text>
        <div className="flex flex-wrap gap-2">
          {EXPRESSION_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              aria-pressed={option.id === expression}
              onClick={() => onExpressionChange(option.id)}
              className={cn(
                CHIP,
                option.id === expression
                  ? "border-zinc-900 bg-zinc-900 text-white"
                  : "border-zinc-200 text-zinc-700 hover:border-zinc-400",
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <Text variant="small" tone="muted" as="h2">
          Preview size
        </Text>
        <div className="flex flex-wrap gap-2">
          {PREVIEW_SIZES.map((size) => (
            <button
              key={size}
              type="button"
              aria-pressed={size === previewSize}
              onClick={() => onPreviewSizeChange(size)}
              className={cn(
                CHIP,
                size === previewSize
                  ? "border-zinc-900 bg-zinc-900 text-white"
                  : "border-zinc-200 text-zinc-700 hover:border-zinc-400",
              )}
            >
              {size}px
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
