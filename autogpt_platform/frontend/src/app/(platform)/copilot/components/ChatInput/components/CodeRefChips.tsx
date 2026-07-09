"use client";

import { CodeIcon, XIcon } from "@/components/atoms/AGPTIcon/icons";
import type { CodeRef } from "../../../store";
import { codeRefLabel } from "../../../helpers/codeRefs";

interface Props {
  refs: CodeRef[];
  onRemove: (id: string) => void;
}

export function CodeRefChips({ refs, onRemove }: Props) {
  if (refs.length === 0) return null;

  return (
    <div className="flex w-full flex-wrap gap-2 px-3 pb-1 pt-2">
      {refs.map((ref) => {
        const label = codeRefLabel(ref);
        return (
          <span
            key={ref.id}
            className="inline-flex items-center gap-1 rounded-full bg-zinc-100 px-3 py-1 text-sm text-zinc-700"
          >
            <CodeIcon className="h-3.5 w-3.5 shrink-0 text-zinc-900" />
            <span className="max-w-[160px] truncate">{label}</span>
            <button
              type="button"
              aria-label={`Remove ${label}`}
              onClick={() => onRemove(ref.id)}
              className="ml-0.5 rounded-full p-0.5 text-zinc-400 transition-colors hover:bg-zinc-200 hover:text-zinc-600"
            >
              <XIcon className="h-3 w-3" weight="bold" />
            </button>
          </span>
        );
      })}
    </div>
  );
}
