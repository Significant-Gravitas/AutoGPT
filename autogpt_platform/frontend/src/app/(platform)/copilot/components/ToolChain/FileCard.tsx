"use client";

import { FileIcon, ImageIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { ChainRow } from "./helpers";
import { CARD, HALF } from "./ResultCards";
import { asObject, formatBytes, str } from "./resultHelpers";
import { TechnicalDetails, TechnicalValue } from "./TechnicalDetails";
import { fileLabel } from "./toolCatalog.shared";
import { KeyValueList } from "./ToolResultViews";

interface Props {
  row: ChainRow;
}

/** A workspace file read, written or deleted: shown by its name, with its
 *  size and a peek at its text. The path is behind technical details. */
export function FileCard({ row }: Props) {
  const input = asObject(row.input);
  const output = asObject(row.output);
  const path =
    (input ? str(input, "file_path", "path", "filename") : null) ??
    (output ? str(output, "path", "name") : null);
  if (!path) return <KeyValueList value={row.output} />;
  const size = output?.size ?? output?.size_bytes;
  const mime = output ? (str(output, "mime_type") ?? "") : "";
  const preview = output ? str(output, "preview", "content_preview") : null;
  const fileIcon = mime.startsWith("image/") ? ImageIcon : FileIcon;
  return (
    <div className="flex flex-col items-start gap-1.5">
      <div className={`${CARD} ${HALF} p-2.5`}>
        <div className="flex items-center gap-2.5">
          <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
            <Icon
              icon={fileIcon}
              size={15}
              className="text-foreground-secondary"
            />
          </div>
          <p className="min-w-0 flex-1 truncate text-sm text-foreground-body">
            {fileLabel(path) ?? "File"}
          </p>
          {typeof size === "number" && (
            <span className="shrink-0 text-xs text-foreground-subtle">
              {formatBytes(size)}
            </span>
          )}
        </div>
        {preview && (
          <p className="mt-1.5 line-clamp-2 whitespace-pre-wrap pl-9 text-xs text-foreground-subtle">
            {preview}
          </p>
        )}
      </div>
      <TechnicalDetails>
        <TechnicalValue>{path}</TechnicalValue>
      </TechnicalDetails>
    </div>
  );
}
