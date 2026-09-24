"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { FieldValue } from "./components/FieldValue";
import { ReferenceValue } from "./components/ReferenceValue";
import {
  type FieldSpec,
  humanize,
  MAX_FIELDS,
  type Reference,
  visibleKeys,
} from "./helpers";

interface Props {
  // Labels and order from the tool's or block's input schema.
  fields: FieldSpec[];
  values: Record<string, unknown>;
  // Keys the server shortened; the approval still binds the whole value.
  clipped?: string[];
  // Keys the card's headline already names.
  hiddenKeys?: string[];
  idsWhenAlone?: boolean;
  // What the id arguments name; one shows as its name, linked when it has a page.
  references?: Reference[];
}

export function ApprovalFields({
  fields,
  values,
  clipped = [],
  hiddenKeys = [],
  idsWhenAlone = false,
  references = [],
}: Props) {
  const [showAll, setShowAll] = useState(false);
  const labels = new Map(fields.map((f) => [f.key, f.label]));
  const ordered = visibleKeys({
    keys: [...fields.map((f) => f.key), ...Object.keys(values)],
    values,
    hiddenKeys,
    idsWhenAlone,
    references,
  }).map((key) => ({ key, label: labels.get(key) ?? humanize(key) }));

  if (ordered.length === 0) return null;
  const shown = showAll ? ordered : ordered.slice(0, MAX_FIELDS);
  const more = ordered.length - shown.length;

  return (
    <div className="flex flex-col gap-2">
      <dl className="grid grid-cols-1 gap-x-4 gap-y-1.5 text-sm sm:grid-cols-[minmax(5rem,9rem)_minmax(0,1fr)] sm:gap-y-2">
        {shown.map((field) => (
          <div key={field.key} className="contents">
            <dt className="text-zinc-500 sm:pt-px">{field.label}</dt>
            <dd className="mb-1.5 min-w-0 text-zinc-900 [overflow-wrap:anywhere] sm:mb-0">
              <FieldOrReference
                name={field.key}
                value={values[field.key]}
                clipped={clipped.includes(field.key)}
                references={references}
              />
            </dd>
          </div>
        ))}
      </dl>
      {more > 0 && (
        <Button
          variant="link"
          size="small"
          className="h-auto min-w-0 self-start px-0 py-0 text-sm"
          onClick={() => setShowAll(true)}
        >
          Show {more} more
        </Button>
      )}
    </div>
  );
}

interface FieldOrReferenceProps {
  name: string;
  value: unknown;
  clipped: boolean;
  references: Reference[];
}

function FieldOrReference({
  name,
  value,
  clipped,
  references,
}: FieldOrReferenceProps) {
  const refs = references.filter((ref) => ref.key === name);
  if (!refs.some((ref) => ref.name))
    return <FieldValue name={name} value={value} clipped={clipped} />;
  const total = Array.isArray(value) ? value.length : 1;
  return <ReferenceValue refs={refs} total={total} />;
}
