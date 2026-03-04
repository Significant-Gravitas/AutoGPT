"use client";

import { cn } from "@/lib/utils";
import {
  CircleNotch as CircleNotchIcon,
  X as XIcon,
} from "@phosphor-icons/react";

interface Props {
  files: File[];
  onRemove: (index: number) => void;
  isUploading?: boolean;
}

export function FileChips({ files, onRemove, isUploading }: Props) {
  if (files.length === 0) return null;

  return (
    <div className="flex w-full flex-wrap gap-2 px-3 pb-2 pt-1">
      {files.map((file, index) => (
        <span
          key={`${file.name}-${file.size}-${index}`}
          className={cn(
            "inline-flex items-center gap-1 rounded-full bg-zinc-100 px-3 py-1 text-sm text-zinc-700",
            isUploading && "opacity-70",
          )}
        >
          <span className="max-w-[160px] truncate">{file.name}</span>
          {isUploading ? (
            <CircleNotchIcon className="ml-0.5 h-3 w-3 animate-spin text-zinc-400" />
          ) : (
            <button
              type="button"
              aria-label={`Remove ${file.name}`}
              onClick={() => onRemove(index)}
              className="ml-0.5 rounded-full p-0.5 text-zinc-400 transition-colors hover:bg-zinc-200 hover:text-zinc-600"
            >
              <XIcon className="h-3 w-3" weight="bold" />
            </button>
          )}
        </span>
      ))}
    </div>
  );
}
