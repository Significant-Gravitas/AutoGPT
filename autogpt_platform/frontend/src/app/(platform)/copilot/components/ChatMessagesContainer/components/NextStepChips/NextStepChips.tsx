"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import type { MessagePart } from "../../helpers";
import { useNextStepChips } from "./useNextStepChips";

interface Props {
  parts: MessagePart[];
}

export function NextStepChips({ parts }: Props) {
  const { suggestions, sentLabel, handleSelect } = useNextStepChips(parts);

  if (suggestions.length === 0) return null;

  return (
    <div
      data-testid="next-step-chips"
      className="mt-3 flex flex-wrap items-center gap-2"
    >
      {suggestions.map((label) => (
        <button
          key={label}
          type="button"
          disabled={sentLabel !== null}
          onClick={() => handleSelect(label)}
          className="group inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-4 py-2 text-sm text-zinc-700 transition-colors hover:border-zinc-300 hover:bg-zinc-50 hover:text-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {label}
          <Icon
            icon={ArrowRight02Icon}
            size={14}
            className="text-zinc-400 transition-colors group-hover:text-zinc-600"
          />
        </button>
      ))}
    </div>
  );
}
