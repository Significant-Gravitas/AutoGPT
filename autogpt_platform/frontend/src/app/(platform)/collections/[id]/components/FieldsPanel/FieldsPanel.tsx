import type { CollectionField } from "@/app/api/__generated__/models/collectionField";
import { FieldType } from "@/app/api/__generated__/models/fieldType";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { useState } from "react";
import { useFieldsPanel } from "./useFieldsPanel";

const TYPE_LABELS: Record<FieldType, string> = {
  text: "Text",
  number: "Number",
  date: "Date",
  select: "Select",
  multi_select: "Multi-select",
  checkbox: "Checkbox",
  url: "Link",
  email: "Email",
};

interface Props {
  collectionID: string;
  fields: CollectionField[];
}

export function FieldsPanel({ collectionID, fields }: Props) {
  const panel = useFieldsPanel(collectionID);
  return (
    <section aria-label="Fields" className="flex flex-col gap-2">
      <Text variant="body-medium" as="h2">
        Fields
      </Text>
      <Text variant="small" tone="secondary">
        A workflow still writing a field&rsquo;s old name writes to the renamed
        one. A type change converts every value, or changes nothing if one would
        not fit.
      </Text>
      <ul className="grid gap-2 sm:grid-cols-2">
        {fields.map((field) => (
          <li key={field.name} className="flex items-center gap-2">
            <FieldName field={field} onRename={panel.rename} />
            <Select
              id={`type-${field.name}`}
              label={`Type of ${field.name}`}
              hideLabel
              size="small"
              wrapperClassName="!mb-0 w-36 shrink-0"
              value={field.type}
              options={Object.values(FieldType).map((type) => ({
                value: type,
                label: TYPE_LABELS[type],
              }))}
              onValueChange={(type) =>
                panel.retype(field.name, type as FieldType)
              }
            />
          </li>
        ))}
      </ul>
    </section>
  );
}

function FieldName({
  field,
  onRename,
}: {
  field: CollectionField;
  onRename(field: string, to: string): void;
}) {
  const [name, setName] = useState(field.name);
  return (
    <Input
      id={`name-${field.name}`}
      label={`Name of ${field.name}`}
      hideLabel
      size="small"
      wrapperClassName="!mb-0 min-w-0 flex-1"
      value={name}
      onChange={(e) => setName(e.target.value)}
      onBlur={() => onRename(field.name, name)}
      onKeyDown={(e) => {
        if (isKey(e, "Enter")) onRename(field.name, name);
        if (isKey(e, "Escape")) setName(field.name);
      }}
    />
  );
}
