"use client";

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { isCodeFile } from "./helpers";

export type FileTypeKey =
  | "pdf"
  | "xls"
  | "json"
  | "img"
  | "html"
  | "video"
  | "react"
  | "code"
  | "generic";

interface FileTypeConfig {
  label: string;
  cardClass: string;
  badgePositionClass: string;
  content: ReactNode;
}

const cardBase =
  "relative h-28 overflow-hidden rounded-md rounded-tr-[15%] bg-white shadow-md shadow-black/[0.065] ring-1 ring-zinc-200";

const DEFAULT_BADGE_POSITION = "bottom-5 -right-2";

function GenericTextContent() {
  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <div className="h-0.5 w-full rounded-full bg-zinc-900/10" />
        <div className="flex gap-1">
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
        </div>
        <div className="flex gap-1">
          <div className="h-0.5 w-1/2 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-1/2 rounded-full bg-zinc-900/10" />
        </div>
        <div className="flex gap-1">
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
        </div>
        <div className="flex gap-1">
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-2/3 rounded-full bg-zinc-900/10" />
        </div>
        <div className="flex gap-1">
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
          <div className="h-0.5 w-1/3 rounded-full bg-zinc-900/10" />
        </div>
      </div>
      <div className="pt-1">
        <div className="h-0.5 w-4 rounded-full bg-zinc-900" />
      </div>
    </div>
  );
}

function XlsContent() {
  return (
    <div className="grid grid-cols-3 gap-px overflow-hidden rounded-sm">
      <div className="col-span-3 grid grid-cols-3 gap-px bg-zinc-900/5">
        <div className="h-2 bg-zinc-900/15" />
        <div className="h-2 bg-zinc-900/15" />
        <div className="h-2 bg-zinc-900/15" />
      </div>
      {Array.from({ length: 21 }).map((_, i) => (
        <div key={i} className="h-2 bg-zinc-900/5" />
      ))}
    </div>
  );
}

function JsonContent() {
  return (
    <div className="space-y-1">
      <div className="font-mono text-[6px] text-zinc-900/40">{"{"}</div>
      <div className="flex items-center gap-1 pl-1.5">
        <div className="h-[3px] w-3 rounded-full bg-zinc-900/30" />
        <div className="text-[5px] text-zinc-900/30">:</div>
        <div className="h-[3px] w-4 rounded-full bg-zinc-900/15" />
      </div>
      <div className="flex items-center gap-1 pl-1.5">
        <div className="h-[3px] w-4 rounded-full bg-zinc-900/30" />
        <div className="text-[5px] text-zinc-900/30">:</div>
        <div className="h-[3px] w-2 rounded-full bg-zinc-900/15" />
      </div>
      <div className="flex items-center gap-1 pl-1.5">
        <div className="h-[3px] w-2.5 rounded-full bg-zinc-900/30" />
        <div className="text-[5px] text-zinc-900/30">:</div>
        <div className="h-[3px] w-5 rounded-full bg-zinc-900/15" />
      </div>
      <div className="flex items-center gap-1 pl-1.5">
        <div className="h-[3px] w-3.5 rounded-full bg-zinc-900/30" />
        <div className="text-[5px] text-zinc-900/30">:</div>
        <div className="h-[3px] w-3 rounded-full bg-zinc-900/15" />
      </div>
      <div className="font-mono text-[6px] text-zinc-900/40">{"}"}</div>
    </div>
  );
}

function HtmlContent() {
  function Tag({
    color,
    w,
    pl,
    closing,
    selfClosing,
  }: {
    color: string;
    w: string;
    pl?: string;
    closing?: boolean;
    selfClosing?: boolean;
  }) {
    return (
      <div className={cn("flex items-center gap-0.5", pl)}>
        <div className="font-mono text-[5px] text-zinc-900/40">
          {closing ? "</" : "<"}
        </div>
        <div className={cn("h-[3px] rounded-full", color, w)} />
        <div className="font-mono text-[5px] text-zinc-900/40">
          {selfClosing ? "/>" : ">"}
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-1">
      <Tag color="bg-zinc-900/30" w="w-3" />
      <Tag color="bg-zinc-900/25" w="w-2.5" pl="pl-1.5" />
      <div className="pl-3">
        <div className="h-[2px] w-6 rounded-full bg-zinc-900/10" />
      </div>
      <Tag color="bg-zinc-900/25" w="w-2.5" pl="pl-1.5" closing />
      <Tag color="bg-zinc-900/20" w="w-2" pl="pl-1.5" selfClosing />
      <Tag color="bg-zinc-900/30" w="w-3" closing />
    </div>
  );
}

function VideoContent() {
  return (
    <div className="relative h-16 overflow-hidden rounded-sm bg-zinc-900/10">
      {/* Play triangle */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <div
          className="h-0 w-0"
          style={{
            borderTop: "5px solid transparent",
            borderBottom: "5px solid transparent",
            borderLeft: "7px solid rgb(24 24 27 / 0.55)",
          }}
        />
      </div>
      {/* Timeline at the bottom */}
      <div className="absolute bottom-1 left-1 right-1 h-1 rounded-full bg-zinc-900/10">
        <div className="h-full w-1/3 rounded-full bg-zinc-900/40" />
      </div>
    </div>
  );
}

function ImgContent() {
  return (
    <div className="relative h-16">
      <div className="absolute bottom-0 left-0 right-0 h-4 bg-gradient-to-t from-zinc-900/20 to-transparent" />
      <div className="absolute bottom-1 left-1 h-3 w-5 rounded-t-full bg-zinc-900/25" />
      <div className="absolute bottom-1 right-2 h-5 w-4 rounded-t-full bg-zinc-900/30" />
      <div className="absolute right-1.5 top-1.5 size-2 rounded-full bg-zinc-900/40" />
    </div>
  );
}

function ReactContent() {
  return (
    <div className="flex h-16 items-center justify-center">
      <svg viewBox="-12 -12 24 24" className="h-10 w-10" aria-hidden>
        <circle r="2" fill="rgb(24 24 27 / 0.55)" />
        <g fill="none" stroke="rgb(24 24 27 / 0.35)" strokeWidth="1">
          <ellipse rx="10" ry="4" />
          <ellipse rx="10" ry="4" transform="rotate(60)" />
          <ellipse rx="10" ry="4" transform="rotate(120)" />
        </g>
      </svg>
    </div>
  );
}

function CodeContent() {
  // Indented "code lines" with a leading angle-bracket motif — visually
  // distinct from the flat GenericTextContent so code reads as code at a glance.
  const lines = [
    { pl: "", w: "w-3" },
    { pl: "pl-2", w: "w-4" },
    { pl: "pl-4", w: "w-2.5" },
    { pl: "pl-4", w: "w-3.5" },
    { pl: "pl-2", w: "w-2" },
    { pl: "", w: "w-3" },
  ];
  return (
    <div className="space-y-1.5">
      {lines.map((line, i) => (
        <div key={i} className={cn("flex items-center gap-1", line.pl)}>
          <div className="font-mono text-[5px] leading-none text-zinc-900/40">
            {i === 0 ? "<" : i === lines.length - 1 ? "/>" : "·"}
          </div>
          <div className={cn("h-[3px] rounded-full bg-zinc-900/20", line.w)} />
        </div>
      ))}
    </div>
  );
}

export const FILE_TYPE_CONFIGS: Record<FileTypeKey, FileTypeConfig> = {
  pdf: {
    label: "PDF",
    cardClass: "w-20 space-y-3 p-3",
    badgePositionClass: "bottom-5 -right-2",
    content: <GenericTextContent />,
  },
  xls: {
    label: "XLS",
    cardClass: "w-20 space-y-2 p-2",
    badgePositionClass: "bottom-5 -right-2",
    content: <XlsContent />,
  },
  json: {
    label: "JSON",
    cardClass: "w-20 space-y-1.5 p-2.5",
    badgePositionClass: "bottom-5 -right-2",
    content: <JsonContent />,
  },
  img: {
    label: "IMG",
    cardClass: "w-20 p-2.5",
    badgePositionClass: "bottom-5 -right-2",
    content: <ImgContent />,
  },
  html: {
    label: "HTML",
    cardClass: "w-20 space-y-1 p-2.5",
    badgePositionClass: "bottom-5 -right-2",
    content: <HtmlContent />,
  },
  video: {
    label: "MP4",
    cardClass: "w-20 p-2.5",
    badgePositionClass: "bottom-5 -right-2",
    content: <VideoContent />,
  },
  react: {
    label: "REACT",
    cardClass: "w-20 p-2.5",
    badgePositionClass: "bottom-5 -right-2",
    content: <ReactContent />,
  },
  code: {
    label: "CODE",
    cardClass: "w-20 space-y-1.5 p-2.5",
    badgePositionClass: "bottom-5 -right-2",
    content: <CodeContent />,
  },
  generic: {
    label: "FILE",
    cardClass: "w-20 space-y-3 p-3",
    badgePositionClass: "bottom-5 -right-2",
    content: <GenericTextContent />,
  },
};

export const FILE_TYPE_KEYS: FileTypeKey[] = [
  "pdf",
  "xls",
  "json",
  "img",
  "html",
  "video",
  "react",
  "code",
  "generic",
];

export function pickFileTypeKey(
  mimeType: string | undefined,
  fileName?: string,
): FileTypeKey {
  const mt = (mimeType ?? "").toLowerCase();
  const name = (fileName ?? "").toLowerCase();
  if (name.endsWith(".jsx") || name.endsWith(".tsx")) return "react";
  // Extension-first for code: `.ts` resolves to `video/mp2t`, so a MIME-first
  // check would render TypeScript as a video.
  if (isCodeFile(name)) return "code";
  if (mt.startsWith("image/")) return "img";
  if (mt.startsWith("video/")) return "video";
  if (mt.includes("pdf")) return "pdf";
  if (mt.includes("html") || mt.includes("xhtml")) return "html";
  if (
    mt.includes("spreadsheet") ||
    mt.includes("excel") ||
    mt.includes("csv")
  ) {
    return "xls";
  }
  if (mt.includes("json")) return "json";
  return "generic";
}

export function deriveBadgeLabel(
  fileName: string | undefined,
  mimeType: string | undefined,
): string {
  const extMatch = (fileName ?? "").match(/\.([a-z0-9]+)$/i);
  if (extMatch) return extMatch[1].toUpperCase().slice(0, 4);
  const sub = (mimeType ?? "").split("/")[1];
  if (sub) return sub.toUpperCase().slice(0, 4);
  return "FILE";
}

interface Props {
  typeKey: FileTypeKey;
  label?: string;
  className?: string;
  size?: "sm" | "md";
  badgeClassName?: string;
}

const SIZE_WRAPPER_CLASS: Record<NonNullable<Props["size"]>, string> = {
  md: "",
  sm: "origin-top-left scale-[0.55] -mr-9 -mb-12",
};

export function FileIllustration({
  typeKey,
  label,
  className,
  size = "md",
  badgeClassName,
}: Props) {
  const config = FILE_TYPE_CONFIGS[typeKey];
  const badgeLabel = label ?? config.label;
  return (
    <div
      className={cn("relative shrink-0", SIZE_WRAPPER_CLASS[size], className)}
    >
      <div className={cn(cardBase, config.cardClass)}>{config.content}</div>
      <span
        className={cn(
          "absolute rounded-md bg-zinc-700 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-white shadow-md shadow-black/10",
          config.badgePositionClass ?? DEFAULT_BADGE_POSITION,
          badgeClassName,
        )}
      >
        {badgeLabel}
      </span>
    </div>
  );
}
