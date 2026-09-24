"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import {
  CLAMP_LINES,
  CODE_MAX_LINES,
  fieldKind,
  humanize,
  lineCount,
  listText,
  scalarText,
} from "../helpers";

interface Props {
  name: string;
  value: unknown;
  clipped: boolean;
}

export function FieldValue({ name, value, clipped }: Props) {
  const kind = fieldKind(name, value);
  const shortened = clipped ? (
    <span className="ml-1 text-zinc-400">(shortened)</span>
  ) : null;

  switch (kind) {
    case "secret":
      return (
        <span className="text-zinc-500">
          <span aria-hidden="true">•••••••• </span>hidden
        </span>
      );
    case "code":
      return (
        <pre
          translate="no"
          className="overflow-auto whitespace-pre-wrap rounded-lg bg-zinc-900 px-3 py-2 font-mono text-[0.8125rem] leading-5 text-zinc-100"
          style={{ maxHeight: `${CODE_MAX_LINES * 1.25 + 1}rem` }}
        >
          {String(value)}
          {shortened}
        </pre>
      );
    case "long":
      return <LongText text={String(value)} shortened={shortened} />;
    case "list":
      return (
        <span translate="no">
          {listText(value as unknown[])}
          {shortened}
        </span>
      );
    case "object":
      return (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 border-l border-zinc-200 pl-3">
          {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-zinc-500">{humanize(k)}</dt>
              <dd className="min-w-0">
                {Array.isArray(v) ? listText(v) : scalarText(v)}
              </dd>
            </div>
          ))}
        </dl>
      );
    case "json":
      return (
        <details className="group">
          <summary className="cursor-pointer text-zinc-600 underline decoration-zinc-300 underline-offset-2 hover:text-zinc-900">
            View as JSON
          </summary>
          <pre className="mt-1.5 max-h-64 overflow-auto rounded-lg bg-zinc-50 px-3 py-2 font-mono text-xs text-zinc-800">
            {JSON.stringify(value, null, 2)}
          </pre>
        </details>
      );
    default:
      return (
        <span>
          {scalarText(value)}
          {shortened}
        </span>
      );
  }
}

interface LongTextProps {
  text: string;
  shortened: React.ReactNode;
}

function LongText({ text, shortened }: LongTextProps) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex flex-col items-start gap-1">
      <div
        className={cn(
          "w-full whitespace-pre-wrap rounded-lg bg-zinc-50 px-3 py-2",
          !open && "line-clamp-3",
        )}
        style={{ WebkitLineClamp: open ? undefined : CLAMP_LINES }}
      >
        {text}
        {shortened}
      </div>
      {!open && (
        <Button
          variant="link"
          size="small"
          className="text-sm"
          onClick={() => setOpen(true)}
        >
          {lineCount(text) > CLAMP_LINES
            ? `Show all ${lineCount(text)} lines`
            : "Show all"}
        </Button>
      )}
    </div>
  );
}
