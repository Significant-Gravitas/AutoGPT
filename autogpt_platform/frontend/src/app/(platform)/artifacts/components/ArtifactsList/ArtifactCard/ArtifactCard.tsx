"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";
import { useFileDrag } from "../../WorkspaceFolders/useFileDrag";
import { FileActionsMenu } from "../FileActionsMenu";
import {
  formatFileSize,
  formatRelativeDate,
  getFileTypeIcon,
  getFileTypeLabel,
} from "../helpers";
import { CardPreview } from "./CardPreview";

interface Props {
  file: WorkspaceFileItem;
  onOpen: (file: WorkspaceFileItem) => void;
}

const CARD_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 8, scale: 0.98, filter: "blur(8px)" },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    filter: "blur(0px)",
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
  },
};

const REDUCED_CARD_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.2 } },
};

export function ArtifactCard({ file, onOpen }: Props) {
  const typeIcon = getFileTypeIcon(file.mime_type, file.name);
  const reduceMotion = useReducedMotion();
  const { handleDragStart, handleDragEnd } = useFileDrag(file.id, file.name);

  return (
    <motion.li
      variants={reduceMotion ? REDUCED_CARD_VARIANTS : CARD_VARIANTS}
      style={{ willChange: "transform, opacity, filter" }}
      className="group relative flex flex-col overflow-hidden rounded-2xl border border-zinc-200 bg-white transition-colors hover:border-zinc-300"
      data-testid="artifacts-list-item"
      draggable
      onDragStartCapture={handleDragStart}
      onDragEndCapture={handleDragEnd}
    >
      {/* Full-card click target: opening the file is the primary action.
          Sits behind the content (z-0); the content is pointer-events-none so
          clicks fall through, except the kebab menu which re-enables them. */}
      <button
        type="button"
        onClick={() => onOpen(file)}
        aria-label={`Open ${file.name}`}
        className="absolute inset-0 z-0 cursor-pointer"
        data-testid="artifacts-card-open"
      />
      <div className="pointer-events-none relative z-10">
        <CardPreview file={file} />
        <div className="flex items-center gap-3 p-3">
          <Icon icon={typeIcon} size={20} className="shrink-0 text-zinc-500" />
          <div className="flex min-w-0 flex-1 flex-col">
            <Text
              variant="body-medium"
              className="truncate text-zinc-900"
              title={file.name}
            >
              {file.name}
            </Text>
            <Text variant="small" className="truncate text-zinc-500">
              {getFileTypeLabel(file.mime_type, file.name)} ·{" "}
              {formatFileSize(file.size_bytes)} ·{" "}
              {formatRelativeDate(file.created_at)}
            </Text>
          </div>
          <div className="pointer-events-auto">
            <FileActionsMenu file={file} />
          </div>
        </div>
      </div>
    </motion.li>
  );
}
