"use client";

import { CopyButton } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/components/CopyButton";
import { visiblePins } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/value";
import { Text } from "@/components/atoms/Text/Text";
import { RunValues } from "./RunValues";

type Values = Record<string, unknown> | null;

interface Props {
  inputs: Values;
  /** Left out while the run is still going. */
  outputs: Values | undefined;
}

/** What went into a run and what came out, each as a table: the name on
 *  the left, the value on the right, drawn to be read. */
export function RunIO({ inputs, outputs }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <ValueTable label="Input" value={inputs} />
      {outputs === undefined ? null : (
        <ValueTable label="Output" value={outputs} />
      )}
    </div>
  );
}

interface TableProps {
  label: string;
  value: Values;
}

function ValueTable({ label, value }: TableProps) {
  const entries = visiblePins(value);
  return (
    <section aria-label={label} className="flex min-w-0 flex-col gap-1.5">
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
      <RunValues value={value} />
    </section>
  );
}
