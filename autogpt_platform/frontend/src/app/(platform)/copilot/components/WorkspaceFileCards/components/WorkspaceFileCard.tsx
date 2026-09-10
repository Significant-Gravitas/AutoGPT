"use client";

import { Delete02Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { classifyArtifact } from "../../ArtifactPanel/helpers";
import { isUploadedFile } from "../../ContextPanel/components/FilesTab/helpers";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";

interface Props {
  file: SessionFile;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

export function WorkspaceFileCard({
  file,
  onOpen,
  onDownload,
  onRequestDelete,
}: Props) {
  const { item } = file;
  const fileIcon = classifyArtifact(item.mime_type ?? null, item.name).icon;
  const canDelete = !isUploadedFile(item);

  return (
    <div className="group relative -mx-2.5 flex items-center gap-3 rounded-xl px-2.5 py-1.5 transition-colors hover:bg-zinc-50">
      <button
        type="button"
        onClick={() => onOpen(file)}
        title={item.name}
        className="flex min-w-0 flex-1 items-center gap-3 text-left"
      >
        <Icon icon={fileIcon} size={18} className="shrink-0 text-zinc-700" />
        {/* Long names fade out rather than ellipsing, so the row keeps a clean
            edge next to the hover actions. */}
        <span className="min-w-0 flex-1 overflow-hidden whitespace-nowrap text-[15px] text-zinc-800 [mask-image:linear-gradient(to_right,black_calc(100%_-_2rem),transparent)]">
          {item.name}
        </span>
      </button>
      {/* Actions stay mounted for keyboard users and only fade in on hover, so
          the row reads as plain text until you reach for it. Touch devices
          never hover, so there they stay visible. */}
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
    </div>
  );
}
