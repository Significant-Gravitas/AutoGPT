import { Button } from "@/components/atoms/Button/Button";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { DownloadSimpleIcon, TrashIcon } from "@phosphor-icons/react";
import { classifyArtifact } from "../../../../ArtifactPanel/helpers";
import {
  formatFileSize,
  formatFileTimestamp,
  isUploadedFile,
} from "../helpers";
import type { SessionFile } from "../useSessionFiles";

interface Props {
  file: SessionFile;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

export function FileRow({ file, onOpen, onDownload, onRequestDelete }: Props) {
  const { item } = file;
  const Icon = classifyArtifact(item.mime_type ?? null, item.name).icon;
  const canDelete = !isUploadedFile(item);

  return (
    <div className="group flex items-center gap-2 rounded-md py-1.5 hover:bg-zinc-50">
      <button
        type="button"
        onClick={() => onOpen(file)}
        className="flex min-w-0 flex-1 flex-col text-left"
        title={item.name}
      >
        <span className="flex min-w-0 items-center gap-2">
          <Icon size={16} className="shrink-0 text-black" />
          <OverflowText
            variant="body"
            value={item.name}
            className="text-zinc-800"
          />
        </span>
        <span className="text-xs text-zinc-400">
          {formatFileSize(item.size_bytes ?? 0)} ·{" "}
          {formatFileTimestamp(item.created_at)}
        </span>
      </button>
      <div className="flex shrink-0 items-center gap-0.5">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onDownload(file)}
              aria-label={`Download ${item.name}`}
            >
              <DownloadSimpleIcon size={16} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Download</TooltipContent>
        </Tooltip>
        {canDelete && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onRequestDelete(file)}
                aria-label={`Delete ${item.name}`}
              >
                <TrashIcon size={16} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Delete</TooltipContent>
          </Tooltip>
        )}
      </div>
    </div>
  );
}
