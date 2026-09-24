"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { FieldValue } from "./components/FieldValue";
import { type FieldSpec, humanize, isShown, MAX_FIELDS } from "./helpers";

interface Props {
  // Labels and order from the tool's or block's input schema.
  fields: FieldSpec[];
  values: Record<string, unknown>;
  // Keys the server shortened; the approval still binds the whole value.
  clipped?: string[];
  // Keys the card's headline already names.
  hiddenKeys?: string[];
}

export function ApprovalFields({
  fields,
  values,
  clipped = [],
  hiddenKeys = [],
}: Props) {
  const [showAll, setShowAll] = useState(false);
  const known = new Set(fields.map((f) => f.key));
  const ordered = [
    ...fields,
    ...Object.keys(values)
      .filter((key) => !known.has(key))
      .map((key) => ({ key, label: humanize(key) })),
  ].filter((field) => isShown(field.key, values[field.key], hiddenKeys));

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
              <FieldValue
                name={field.key}
                value={values[field.key]}
                clipped={clipped.includes(field.key)}
              />
            </dd>
          </div>
        ))}
      </dl>
      {more > 0 && (
        <Button
          variant="link"
          size="small"
          className="self-start text-sm"
          onClick={() => setShowAll(true)}
        >
          Show {more} more
        </Button>
      )}
    </div>
  );
}
