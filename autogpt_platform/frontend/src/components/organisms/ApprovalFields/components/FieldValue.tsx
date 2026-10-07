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
  REDACTED,
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
    <span className="ml-1 text-muted-foreground">(shortened)</span>
  ) : null;

  switch (kind) {
    case "secret":
      return <Hidden />;
    case "code":
      return (
        <pre
          translate="no"
          className="overflow-auto rounded-lg bg-zinc-900 px-3 py-2 font-mono text-[0.8125rem] leading-5 whitespace-pre-wrap text-zinc-100"
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
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 border-l border-border pl-3">
          {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted-foreground">{humanize(k)}</dt>
              <dd className="min-w-0">
                {v === REDACTED ? (
                  <Hidden />
                ) : Array.isArray(v) ? (
                  listText(v)
                ) : (
                  scalarText(v)
                )}
              </dd>
            </div>
          ))}
        </dl>
      );
    case "json":
      return (
        <details className="group">
          <summary className="cursor-pointer text-muted-foreground underline decoration-zinc-300 underline-offset-2 hover:text-foreground">
            View as JSON
          </summary>
          <pre className="mt-1.5 max-h-64 overflow-auto rounded-lg bg-muted/50 px-3 py-2 font-mono text-xs text-foreground">
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
        tabIndex={open ? 0 : undefined}
        className={cn(
          "w-full rounded-lg bg-muted/50 px-3 py-2",
          open &&
            "max-h-96 overflow-y-auto focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-hidden",
        )}
      >
        <p
          className={cn("whitespace-pre-wrap", !open && "line-clamp-3")}
          style={{ WebkitLineClamp: open ? undefined : CLAMP_LINES }}
        >
          {text}
          {shortened}
        </p>
      </div>
      {!open && (
        <Button
          variant="link"
          size="md"
          className="h-auto min-w-0 px-0 py-0 text-sm"
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

function Hidden() {
  return (
    <span className="text-muted-foreground">
      <span aria-hidden="true">•••••••• </span>hidden
    </span>
  );
}
