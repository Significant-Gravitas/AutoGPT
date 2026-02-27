import { Paperclip as PaperclipIcon } from "@phosphor-icons/react";
import type { FileUIPart } from "ai";

interface Props {
  files: FileUIPart[];
}

export function MessageAttachments({ files }: Props) {
  if (files.length === 0) return null;

  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {files.map((file, i) => (
        <span
          key={`${file.filename}-${i}`}
          className="inline-flex items-center gap-1 rounded-md bg-purple-200/50 px-2 py-0.5 text-xs text-purple-700/80"
        >
          <PaperclipIcon className="h-3 w-3 shrink-0" />
          <span className="max-w-[140px] truncate">
            {file.filename || "file"}
          </span>
        </span>
      ))}
    </div>
  );
}
