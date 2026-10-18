"use client";

import { Delete02Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { isUploadedFile } from "../../ContextPanel/components/FilesTab/helpers";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";

interface Props {
  file: SessionFile;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

/** Actions stay mounted for keyboard users and only fade in on hover, so the
 *  row reads as plain text until you reach for it. Touch devices never hover,
 *  so there they stay visible. */
export function FileRowActions({ file, onDownload, onRequestDelete }: Props) {
  const { item } = file;
  const canDelete = !isUploadedFile(item);

  return (
    <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
      <Button
        variant="ghost"
        size="icon"
        onClick={() => onDownload(file)}
        aria-label={`Download ${item.name}`}
        className="size-7 rounded-lg !p-0 text-zinc-500"
      >
        <Icon icon={Download01Icon} size={14} />
      </Button>
      {canDelete && (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onRequestDelete(file)}
          aria-label={`Delete ${item.name}`}
          className="size-7 rounded-lg !p-0 text-zinc-500"
        >
          <Icon icon={Delete02Icon} size={14} />
        </Button>
      )}
    </div>
  );
}
