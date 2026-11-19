"use client";

import { Text } from "@/components/atoms/Text/Text";
import type { TraceStep } from "../trace";
import { labelOf, unwrap, visiblePins } from "../value";
import { CopyButton } from "./CopyButton";
import { ValueView } from "./ValueView";

interface Props {
  step: TraceStep;
}

/** What went into a block run and what came out of it, with the error when
 *  it failed. Values read as text, links, lists and fields; each side still
 *  copies as JSON. */
export function StepDetail({ step }: Props) {
  const outputs = withoutError(step.output);
  return (
    <div className="flex flex-col gap-2.5 px-2 pt-0.5 pb-2.5 pl-8">
      {step.error ? (
        <div role="alert" className="rounded-lg bg-error-surface px-2.5 py-2">
          <Text
            variant="small"
            as="p"
            tone="error"
            className="break-words whitespace-pre-wrap"
          >
            {step.error}
          </Text>
        </div>
      ) : null}
      <ValueSection label="In" value={step.input} />
      <ValueSection label="Out" value={outputs} />
    </div>
  );
}

function withoutError(output: TraceStep["output"]) {
  if (!output) return null;
  const entries = Object.entries(output).filter(([key]) => key !== "error");
  return entries.length ? Object.fromEntries(entries) : null;
}

interface SectionProps {
  label: string;
  value: Record<string, unknown> | null;
}

/** A named set of values, such as a run's input, drawn to be read; the
 *  button copies them as JSON. */
export function ValueSection({ label, value }: SectionProps) {
  const entries = visiblePins(value);
  return (
    <section aria-label={label} className="flex flex-col gap-1">
      <div className="flex h-6 items-center justify-between">
        <Text variant="small-medium" as="h4" tone="quiet">
          {label}
        </Text>
        {entries.length ? (
          <CopyButton
            text={JSON.stringify(value, null, 2)}
            label={`Copy ${label.toLowerCase()}`}
          />
        ) : null}
      </div>
      {entries.length === 0 ? (
        <Text variant="small" as="p" tone="faint">
          Nothing
        </Text>
      ) : (
        <dl className="flex flex-col gap-3 rounded-lg bg-surface-tray p-2.5">
          {entries.map(([key, item]) => (
            <div key={key} className="flex min-w-0 flex-col gap-1">
              <Text variant="small-medium" as="dt" tone="quiet">
                {labelOf(key)}
              </Text>
              <dd className="min-w-0">
                <ValueView value={unwrap(item)} />
              </dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
