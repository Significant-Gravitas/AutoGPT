"use client";

import {
  CheckmarkCircle02Icon,
  CircleIcon,
  FileIcon,
  GlobeIcon,
  ImageIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useLayoutEffect, useRef, useState } from "react";
import type { ChainRow } from "./helpers";
import { CARD, HALF } from "./ResultCards";
import {
  asItems,
  asObject,
  formatBytes,
  humanizeKey,
  inline,
  resultItemKey,
  safeHostname,
  str,
} from "./resultHelpers";

interface SearchResultsProps {
  items: Record<string, unknown>[];
  answer?: string | null;
}

interface RowProps {
  row: ChainRow;
}

interface ItemsProps {
  items: Record<string, unknown>[];
}

interface ValueProps {
  value: unknown;
}

function ResultFavicon({ domain }: { domain: string | null }) {
  const [failed, setFailed] = useState(false);
  if (!domain || failed) {
    return (
      <Icon icon={GlobeIcon} size={14} className="shrink-0 text-zinc-400" />
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`https://${encodeURIComponent(domain)}/favicon.ico`}
      alt=""
      width={14}
      height={14}
      className="size-3.5 shrink-0 rounded-[3px]"
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
    />
  );
}

function ClampedAnswer({ answer }: { answer: string }) {
  const textRef = useRef<HTMLParagraphElement>(null);
  const [clamped, setClamped] = useState(false);

  useLayoutEffect(() => {
    const el = textRef.current;
    if (el) setClamped(el.scrollHeight > el.clientHeight + 1);
  }, [answer]);

  return (
    <div className="relative">
      <p
        ref={textRef}
        className="line-clamp-3 px-3 py-2 text-[13px] leading-5 text-zinc-600"
      >
        {answer}
      </p>
      {clamped && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-white to-transparent" />
      )}
    </div>
  );
}

export function SearchResults({ items, answer }: SearchResultsProps) {
  return (
    <div className={CARD + " divide-y divide-zinc-100"}>
      {answer && <ClampedAnswer answer={answer} />}
      {items.map((item, i) => {
        const url = str(item, "url", "link");
        const domain = url ? safeHostname(url) : null;
        const title = str(item, "title", "snippet") ?? inline(item);
        return (
          <div
            key={resultItemKey(item, i)}
            className="flex items-center gap-2.5 px-3 py-2"
          >
            <ResultFavicon domain={domain} />
            {url ? (
              <a
                href={url}
                target="_blank"
                rel="noreferrer"
                className="min-w-0 truncate text-[13px] text-zinc-700 hover:text-zinc-900 hover:underline"
              >
                {title}
              </a>
            ) : (
              <p className="min-w-0 truncate text-[13px] text-zinc-700">
                {title}
              </p>
            )}
            {domain && (
              <span className="ml-auto shrink-0 text-xs text-zinc-400">
                {domain}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function Terminal({ row }: RowProps) {
  const input = asObject(row.input);
  const output = asObject(row.output);
  const command = input ? str(input, "command") : null;
  const stdout = output ? str(output, "stdout", "stderr") : null;
  const exitCode = output?.exit_code;
  return (
    <div className="rounded-xl bg-zinc-900 p-3 font-mono text-[11px] leading-4">
      {command && (
        <p className="whitespace-pre-wrap break-words text-zinc-400">
          <span className="select-none text-zinc-500">$ </span>
          {command}
        </p>
      )}
      {stdout && (
        <pre className="mt-1.5 max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-zinc-100 scrollbar-none">
          {stdout}
        </pre>
      )}
      {typeof exitCode === "number" && exitCode !== 0 && (
        <p className="mt-1.5 text-red-400">exit {exitCode}</p>
      )}
    </div>
  );
}

export function TodoList({ row }: RowProps) {
  const input = asObject(row.input);
  const todos = input ? asItems(input.todos) : null;
  if (!todos) return <KeyValueList value={row.output} />;
  return (
    <div className={CARD + " flex flex-col gap-1.5 p-3"}>
      {todos.map((todo, i) => {
        const done = todo.status === "completed";
        const active = todo.status === "in_progress";
        return (
          <div
            key={resultItemKey(todo, i)}
            className="flex items-center gap-2 text-[13px]"
          >
            {done ? (
              <Icon
                icon={CheckmarkCircle02Icon}
                size={15}
                className="shrink-0 text-green-500"
              />
            ) : (
              <Icon
                icon={CircleIcon}
                size={15}
                className={
                  "shrink-0 " + (active ? "text-purple-500" : "text-zinc-300")
                }
              />
            )}
            <span
              className={
                done
                  ? "text-zinc-400 line-through"
                  : active
                    ? "font-medium text-zinc-800"
                    : "text-zinc-600"
              }
            >
              {str(todo, "content", "activeForm") ?? inline(todo)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function FileCard({ row }: RowProps) {
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
    <div className={`${CARD} ${HALF} p-2.5`}>
      <div className="flex items-center gap-2.5">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
          <Icon icon={fileIcon} size={15} className="text-zinc-600" />
        </div>
        <p className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-700">
          {path}
        </p>
        {typeof size === "number" && (
          <span className="shrink-0 text-xs text-zinc-400">
            {formatBytes(size)}
          </span>
        )}
      </div>
      {preview && (
        <p className="mt-1.5 line-clamp-2 whitespace-pre-wrap pl-9 text-xs text-zinc-400">
          {preview}
        </p>
      )}
    </div>
  );
}

export function OutputList({ items }: ItemsProps) {
  return (
    <div className={`${CARD} ${HALF} divide-y divide-zinc-100`}>
      {items.map((item, i) => (
        <div key={resultItemKey(item, i)} className="p-2.5">
          <p className="text-[11px] uppercase tracking-wide text-zinc-400">
            {str(item, "name", "key", "label") ?? `Output ${i + 1}`}
          </p>
          <p className="mt-0.5 whitespace-pre-wrap break-words text-[13px] text-zinc-800">
            {inline(item.value ?? item)}
          </p>
        </div>
      ))}
    </div>
  );
}

export function KeyValueList({ value }: ValueProps) {
  const obj = asObject(value);
  if (!obj || Object.keys(obj).length === 0) {
    if (typeof value !== "string" || !value.trim()) return null;
    return (
      <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-zinc-50 p-2.5 font-mono text-[11px] leading-4 text-zinc-600 scrollbar-none">
        {value}
      </pre>
    );
  }
  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      {Object.entries(obj).map(([key, entryValue]) => (
        <div
          key={key}
          className="flex items-baseline gap-3 py-1 text-xs first:pt-0 last:pb-0"
        >
          <span className="shrink-0 text-zinc-400">{humanizeKey(key)}</span>
          <span className="ml-auto min-w-0 break-words text-right font-medium text-zinc-700">
            {inline(entryValue)}
          </span>
        </div>
      ))}
    </div>
  );
}
