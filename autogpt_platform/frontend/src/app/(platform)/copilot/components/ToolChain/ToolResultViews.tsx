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
  displayValue,
  formatBytes,
  isBlankOutput,
  NOTHING_FOUND,
  humanizeKey,
  inline,
  parseTruncatedOutput,
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
      <Icon
        icon={GlobeIcon}
        size={14}
        className="shrink-0 text-foreground-subtle"
      />
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
        className="line-clamp-3 px-3 py-2 text-body-sm leading-5 text-foreground-secondary"
      >
        {answer}
      </p>
      {clamped && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-surface to-transparent" />
      )}
    </div>
  );
}

export function SearchResults({ items, answer }: SearchResultsProps) {
  return (
    <div className={CARD + " divide-y divide-separator"}>
      {answer && <ClampedAnswer answer={answer} />}
      {items.map((item, i) => {
        const url = str(item, "url", "link");
        const domain = url ? safeHostname(url) : null;
        const title = str(item, "title", "snippet") ?? inline(item);
        const citation = typeof item.n === "number" ? item.n : null;
        return (
          <div
            key={citation === null ? resultItemKey(item, i) : `n:${citation}`}
            className="flex items-center gap-2.5 px-3 py-2"
          >
            {citation !== null && (
              <span className="shrink-0 text-xs tabular-nums text-foreground-secondary">
                [{citation}]
              </span>
            )}
            <ResultFavicon domain={domain} />
            {url ? (
              <a
                href={url}
                target="_blank"
                rel="noreferrer"
                className="min-w-0 truncate text-body-sm leading-inherit text-foreground-body hover:text-foreground-strong hover:underline"
              >
                {title}
              </a>
            ) : (
              <p className="min-w-0 truncate text-body-sm leading-inherit text-foreground-body">
                {title}
              </p>
            )}
            {domain && (
              <span className="ml-auto shrink-0 text-xs text-foreground-subtle">
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
    <div className="rounded-xl bg-action-hover p-3 font-mono text-caption-sm leading-4">
      {command && (
        <p className="whitespace-pre-wrap break-words text-foreground-subtle">
          <span className="select-none text-foreground-tertiary">$ </span>
          {command}
        </p>
      )}
      {stdout && (
        <pre
          // TODO(tokens): no fitting role for text-zinc-100 #EFEFF0; #EFEFF0 is surface-hover (a fill) and separator (a line colour)
          className="mt-1.5 max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-zinc-100 scrollbar-none"
        >
          {stdout}
        </pre>
      )}
      {typeof exitCode === "number" && exitCode !== 0 && (
        <p
          // TODO(tokens): no exact role for text-red-400
          className="mt-1.5 text-red-400"
        >
          exit {exitCode}
        </p>
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
            className="flex items-center gap-2 text-body-sm leading-inherit"
          >
            {done ? (
              <Icon
                icon={CheckmarkCircle02Icon}
                size={15}
                // TODO(tokens): no exact role for text-green-500
                className="shrink-0 text-green-500"
              />
            ) : (
              <Icon
                icon={CircleIcon}
                size={15}
                className={
                  "shrink-0 " +
                  (active ? "text-brand" : "text-foreground-disabled")
                }
              />
            )}
            <span
              className={
                done
                  ? "text-foreground-subtle line-through"
                  : active
                    ? "font-medium text-foreground-emphasis"
                    : "text-foreground-secondary"
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
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
          <Icon
            icon={fileIcon}
            size={15}
            className="text-foreground-secondary"
          />
        </div>
        <p className="min-w-0 flex-1 truncate font-mono text-xs text-foreground-body">
          {path}
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
  );
}

export function OutputList({ items }: ItemsProps) {
  return (
    <div className={`${CARD} ${HALF} divide-y divide-separator`}>
      {items.map((item, i) => (
        <div key={resultItemKey(item, i)} className="p-2.5">
          <p className="text-caption-sm uppercase leading-inherit tracking-wide text-foreground-subtle">
            {str(item, "name", "key", "label") ?? `Output ${i + 1}`}
          </p>
          <p className="mt-0.5 whitespace-pre-wrap break-words text-body-sm leading-inherit text-foreground-emphasis">
            {displayValue(item.value ?? item)}
          </p>
        </div>
      ))}
    </div>
  );
}

export function KeyValueList({ value }: ValueProps) {
  if (isBlankOutput(value)) return <KeyValueList value={NOTHING_FOUND} />;
  const obj = asObject(value);
  if (!obj || Object.keys(obj).length === 0) {
    if (typeof value !== "string" || !value.trim()) return null;
    const truncated = parseTruncatedOutput(value);
    return (
      <div className={`${CARD} w-full rounded-2xl p-3`}>
        <div className="relative">
          <p className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-body-sm leading-relaxed text-foreground-secondary scrollbar-thin scrollbar-track-transparent scrollbar-thumb-surface-active">
            {truncated?.preview ?? value}
          </p>
          {truncated && (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-surface to-transparent" />
          )}
        </div>
        {truncated && (
          <p className="mt-2 text-caption-sm leading-inherit text-foreground-subtle">
            Preview of {truncated.totalChars.toLocaleString()} characters — full
            output saved to the workspace
          </p>
        )}
      </div>
    );
  }
  const entries = Object.entries(obj).filter(
    ([, entryValue]) => entryValue !== null && entryValue !== undefined,
  );
  if (entries.length === 0) return <KeyValueList value={NOTHING_FOUND} />;
  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      {entries.map(([key, entryValue]) => (
        <div
          key={key}
          className="flex items-baseline gap-3 py-1 text-xs first:pt-0 last:pb-0"
        >
          <span className="shrink-0 text-foreground-subtle">
            {humanizeKey(key)}
          </span>
          <span className="ml-auto min-w-0 break-words text-right font-medium text-foreground-body">
            {displayValue(entryValue)}
          </span>
        </div>
      ))}
    </div>
  );
}
