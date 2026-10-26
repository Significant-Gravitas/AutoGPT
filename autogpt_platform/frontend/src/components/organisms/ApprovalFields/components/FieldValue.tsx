"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import {
  CLAMP_LINES,
  fieldKind,
  formattedText,
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
  format?: string | null;
}

export function FieldValue({ name, value, clipped, format }: Props) {
  const kind = fieldKind(name, value);
  const shortened = clipped ? (
    <span
      data-slot="field-value-shortened"
      className="ml-1 text-foreground-subtle"
    >
      (shortened)
    </span>
  ) : null;
  const formatted = formattedText(format, value);
  if (formatted)
    return <span data-slot="field-value-formatted">{formatted}</span>;

  switch (kind) {
    case "secret":
      return <Hidden />;
    case "code":
      return (
        <pre
          translate="no"
          data-slot="field-value-code"
          // max-h-64: twelve lines at leading-5 plus py-2.
          // TODO(tokens): no foreground role for zinc-100 (#EFEFF0) on the
          // dark code surface.
          className="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-action-hover px-3 py-2 font-mono text-body-sm leading-5 text-zinc-100"
        >
          {String(value)}
          {shortened}
        </pre>
      );
    case "long":
      return <LongText text={String(value)} shortened={shortened} />;
    case "list":
      return (
        <span data-slot="field-value-list" translate="no">
          {listText(value as unknown[])}
          {shortened}
        </span>
      );
    case "object":
      return (
        <dl
          data-slot="field-value-object"
          className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 border-l border-border-base pl-3"
        >
          {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <div
              key={k}
              data-slot="field-value-object-row"
              className="contents"
            >
              <dt
                data-slot="field-value-object-label"
                className="text-foreground-tertiary"
              >
                {humanize(k)}
              </dt>
              <dd data-slot="field-value-object-value" className="min-w-0">
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
        <details data-slot="field-value-json" className="group">
          <summary
            data-slot="field-value-json-toggle"
            className="cursor-pointer text-foreground-secondary underline decoration-border-strong underline-offset-2 hover:text-foreground-strong"
          >
            View as JSON
          </summary>
          <pre
            data-slot="field-value-json-body"
            className="mt-1.5 max-h-64 overflow-auto rounded-lg bg-surface-muted px-3 py-2 font-mono text-xs text-foreground-emphasis"
          >
            {JSON.stringify(value, null, 2)}
          </pre>
        </details>
      );
    default:
      return (
        <span data-slot="field-value-scalar">
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
    <div
      data-slot="field-value-long-text"
      className="flex flex-col items-start gap-1"
    >
      <div
        tabIndex={open ? 0 : undefined}
        data-slot="field-value-long-text-body"
        className={cn(
          "w-full rounded-lg bg-surface-muted px-3 py-2",
          open &&
            "max-h-96 overflow-y-auto focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border-strong",
        )}
      >
        <p
          data-slot="field-value-long-text-content"
          // line-clamp-3 is CLAMP_LINES.
          className={cn("whitespace-pre-wrap", !open && "line-clamp-3")}
        >
          {text}
          {shortened}
        </p>
      </div>
      {!open && (
        <Button
          variant="link"
          size="small"
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
    <span data-slot="field-value-hidden" className="text-foreground-tertiary">
      <span data-slot="field-value-hidden-mask" aria-hidden="true">
        ••••••••{" "}
      </span>
      hidden
    </span>
  );
}
