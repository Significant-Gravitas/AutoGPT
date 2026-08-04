"use client";

import {
  ArrowSquareOutIcon,
  CheckCircleIcon,
  GlobeIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

export function Favicon({ url, size = 14 }: { url: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  let domain: string | null = null;
  try {
    domain = new URL(url).hostname;
  } catch {}
  if (!domain || failed)
    return <GlobeIcon size={size} className="shrink-0 text-zinc-400" />;
  return (
    <Image
      src={`https://www.google.com/s2/favicons?domain=${domain}&sz=32`}
      alt=""
      width={size}
      height={size}
      className="shrink-0 rounded-sm object-contain"
      onError={() => setFailed(true)}
    />
  );
}

export const CARD = "rounded-xl bg-white ring-1 ring-zinc-200/70";
export const HALF = "w-full sm:w-1/2";

const STATUS_STYLES: Record<string, string> = {
  COMPLETED: "bg-green-50 text-green-600",
  FAILED: "bg-red-50 text-red-500",
  RUNNING: "bg-purple-50 text-purple-600",
  QUEUED: "bg-amber-50 text-amber-600",
};

export function StatusPill({ status }: { status: string }) {
  const normalized = status.toUpperCase();
  return (
    <span
      className={
        "shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium " +
        (STATUS_STYLES[normalized] ?? "bg-zinc-100 text-zinc-500")
      }
    >
      {normalized.toLowerCase()}
    </span>
  );
}

export function StatusCard({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2 p-2.5`}>
      {ok ? (
        <CheckCircleIcon
          size={16}
          weight="fill"
          className="shrink-0 text-green-500"
        />
      ) : (
        <XCircleIcon
          size={16}
          weight="fill"
          className="shrink-0 text-red-400"
        />
      )}
      <p className="min-w-0 truncate text-[13px] font-medium text-zinc-700">
        {label}
      </p>
    </div>
  );
}

export function StatCard({ value, label }: { value: number; label: string }) {
  return (
    <div className={`${CARD} ${HALF} flex items-baseline gap-2 p-2.5`}>
      <span className="text-lg font-semibold leading-none text-zinc-800">
        {value.toLocaleString()}
      </span>
      <span className="min-w-0 truncate text-xs text-zinc-500">{label}</span>
    </div>
  );
}

export function ChipList({ label, items }: { label: string; items: string[] }) {
  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      <p className="mb-1.5 text-[11px] uppercase tracking-wide text-zinc-400">
        {label}
      </p>
      <div className="flex flex-wrap gap-1">
        {items.map((item, i) => (
          <span
            key={i}
            className="max-w-full truncate rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600"
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

export function LinkCard({
  url,
  title,
  meta,
}: {
  url: string;
  title?: string;
  meta?: string;
}) {
  let domain = url;
  try {
    domain = new URL(url).hostname.replace(/^www\./, "");
  } catch {}

  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Favicon url={url} size={15} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-800">
          {title ?? domain}
        </p>
        <p className="truncate text-xs text-zinc-500">{title ? domain : url}</p>
      </div>
      {meta && <span className="shrink-0 text-xs text-zinc-400">{meta}</span>}
      <Link
        href={url}
        target="_blank"
        rel="noreferrer"
        aria-label="Open link"
        className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
      >
        <ArrowSquareOutIcon size={14} />
      </Link>
    </div>
  );
}
