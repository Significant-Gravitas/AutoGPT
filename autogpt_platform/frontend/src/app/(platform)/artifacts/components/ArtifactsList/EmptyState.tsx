"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PlusSignIcon, Upload03Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import { useNewMenu } from "../NewMenu/useNewMenu";
import type { EmptyStateContent } from "./helpers";

const EASE_OUT_QUINT = [0.22, 1, 0.36, 1] as const;

interface Props {
  content: EmptyStateContent;
  // When folders are already listed, a full empty state would sit right
  // under them and read as contradictory; show a single quiet line.
  compact?: boolean;
}

export function EmptyState({ content, compact = false }: Props) {
  const shouldReduceMotion = useReducedMotion();
  const { fileInputRef, isUploading, openFilePicker, handleFilesSelected } =
    useNewMenu(content.uploadFolderId);

  if (compact) {
    return (
      <Text
        variant="body"
        className="px-2 py-6 text-zinc-500"
        data-testid="artifacts-empty"
      >
        {content.title}
      </Text>
    );
  }

  function fadeUp(delay: number) {
    if (shouldReduceMotion) {
      return {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        transition: { duration: 0.2, delay: 0 },
      };
    }
    return {
      initial: { opacity: 0, y: 8 },
      animate: { opacity: 1, y: 0 },
      transition: { duration: 0.35, ease: EASE_OUT_QUINT, delay },
    };
  }

  const hasActions = content.showUpload || content.chatHref !== null;

  return (
    <div
      className="flex flex-col items-center justify-center gap-8 px-6 py-16 text-center"
      data-testid="artifacts-empty"
    >
      <motion.div {...fadeUp(0)}>
        <StackedFilesIllustration />
      </motion.div>

      <div className="flex max-w-md flex-col items-center gap-2">
        <motion.div {...fadeUp(0.28)}>
          <Text variant="h3" className="text-zinc-900">
            {content.title}
          </Text>
        </motion.div>
        <motion.div {...fadeUp(0.36)}>
          <Text variant="body" className="text-zinc-500">
            {content.description}
          </Text>
        </motion.div>
      </div>

      {hasActions ? (
        <div className="flex flex-col items-center gap-3 sm:flex-row">
          {content.showUpload ? (
            <motion.div {...fadeUp(0.44)}>
              <Button
                variant="primary"
                size="large"
                onClick={openFilePicker}
                disabled={isUploading}
                leftIcon={<Icon icon={Upload03Icon} className="h-4 w-4" />}
              >
                {isUploading ? "Uploading…" : "Upload a file"}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                tabIndex={-1}
                aria-hidden
                onChange={handleFilesSelected}
                data-testid="artifacts-empty-upload-input"
              />
            </motion.div>
          ) : null}
          {content.chatHref ? (
            <motion.div {...fadeUp(0.5)}>
              <Button
                as="NextLink"
                href={content.chatHref}
                variant={content.showUpload ? "secondary" : "primary"}
                size="large"
                leftIcon={<Icon icon={PlusSignIcon} className="h-4 w-4" />}
              >
                {content.chatLabel}
              </Button>
            </motion.div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

// Three fanned documents, drawn in the same palette as the library's
// stacked cards so the two empty states read as one family.
const SHEETS = [
  { rotate: -10, x: 88, opacity: 0.55 },
  { rotate: 8, x: 176, opacity: 0.8 },
  { rotate: 0, x: 132, opacity: 1 },
];

function StackedFilesIllustration() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <svg
      width="320"
      height="160"
      viewBox="0 0 320 160"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className="select-none"
    >
      {SHEETS.map((sheet, i) => (
        <motion.g
          key={i}
          initial={
            shouldReduceMotion
              ? { opacity: 0 }
              : { opacity: 0, y: 12, scale: 0.97 }
          }
          animate={
            shouldReduceMotion
              ? { opacity: sheet.opacity }
              : { opacity: sheet.opacity, y: 0, scale: 1 }
          }
          transition={{
            duration: shouldReduceMotion ? 0.2 : 0.45,
            ease: EASE_OUT_QUINT,
            delay: shouldReduceMotion ? 0 : i * 0.08,
          }}
          style={{ transformOrigin: "160px 96px", transformBox: "fill-box" }}
        >
          <Sheet x={sheet.x} rotate={sheet.rotate} />
        </motion.g>
      ))}
    </svg>
  );
}

function Sheet({ x, rotate }: { x: number; rotate: number }) {
  const width = 100;
  const height = 124;
  const y = 18;
  const fold = 22;
  const padding = 16;
  const cx = x + width / 2;
  const cy = y + height / 2;

  return (
    <g transform={`rotate(${rotate} ${cx} ${cy})`}>
      <path
        d={`M${x + 12} ${y}H${x + width - fold}L${x + width} ${y + fold}V${y + height - 12}a12 12 0 0 1-12 12H${x + 12}a12 12 0 0 1-12-12V${y + 12}a12 12 0 0 1 12-12Z`}
        fill="white"
        stroke="#E4E4E7"
        strokeWidth={1}
      />
      <path
        d={`M${x + width - fold} ${y}V${y + fold}H${x + width}`}
        fill="#F4F4F5"
        stroke="#E4E4E7"
        strokeWidth={1}
      />
      {[0, 1, 2, 3].map((line) => (
        <rect
          key={line}
          x={x + padding}
          y={y + 40 + line * 16}
          width={line === 3 ? width * 0.4 : width - padding * 2}
          height={6}
          rx={3}
          fill="#E4E4E7"
        />
      ))}
    </g>
  );
}
