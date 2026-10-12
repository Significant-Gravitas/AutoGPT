"use client";

import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";

interface Props {
  brief: string;
  isEditing: boolean;
  draft: string;
  onDraftChange: (value: string) => void;
  isExpanded: boolean;
  onToggleExpanded: () => void;
}

const LONG_BRIEF = 360;

export function HandoffBrief({
  brief,
  isEditing,
  draft,
  onDraftChange,
  isExpanded,
  onToggleExpanded,
}: Props) {
  if (isEditing) {
    return (
      <Input
        id="handoff-brief-edit"
        label="Brief"
        hideLabel
        type="textarea"
        rows={6}
        size="small"
        value={draft}
        onChange={(e) => onDraftChange(e.target.value)}
        wrapperClassName="mb-0"
      />
    );
  }
  const isLong = brief.length > LONG_BRIEF || brief.split("\n").length > 6;
  return (
    <div className="flex flex-col items-start gap-1">
      <p
        className={cn(
          "whitespace-pre-wrap text-sm leading-[22px] text-zinc-900",
          !isExpanded && "line-clamp-6",
        )}
      >
        {brief}
      </p>
      {isLong && (
        <button
          type="button"
          onClick={onToggleExpanded}
          className="text-xs text-zinc-600 underline underline-offset-2 hover:text-zinc-900"
        >
          {isExpanded ? "Show less" : "Show all"}
        </button>
      )}
    </div>
  );
}
