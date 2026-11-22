import { Button } from "@/components/atoms/Button/Button";
import { Checkbox } from "@/components/atoms/Checkbox/Checkbox";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { inputType, parseInput, type TableColumn } from "../helpers";

interface Props {
  columns: TableColumn[];
  onAdd(values: Record<string, unknown>): Promise<boolean>;
  onCancel(): void;
}

export function NewRowForm({ columns, onAdd, onCancel }: Props) {
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [checks, setChecks] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const values: Record<string, unknown> = { ...checks };
    for (const column of columns) {
      const parsed = parseInput(column.type, texts[column.field] ?? "");
      if (parsed !== null) values[column.field] = parsed;
    }
    setSaving(true);
    const added = await onAdd(values);
    setSaving(false);
    if (added) onCancel();
  }

  return (
    <form
      onSubmit={submit}
      aria-label="New row"
      className="flex flex-col gap-3 rounded-xl bg-surface-muted p-3"
    >
      <Text variant="small-medium">New row</Text>
      <div className="grid gap-2 sm:grid-cols-2">
        {columns.map((column) =>
          column.type === "checkbox" ? (
            <label
              key={column.field}
              className="flex items-center gap-2 self-end py-2"
            >
              <Checkbox
                checked={checks[column.field] ?? false}
                onCheckedChange={(checked) =>
                  setChecks((c) => ({ ...c, [column.field]: checked === true }))
                }
              />
              <Text as="span" variant="body">
                {column.field}
              </Text>
            </label>
          ) : (
            <Input
              key={column.field}
              id={`new-row-${column.field}`}
              label={column.field}
              size="small"
              type={inputType(column.type)}
              value={texts[column.field] ?? ""}
              onChange={(e) =>
                setTexts((t) => ({ ...t, [column.field]: e.target.value }))
              }
            />
          ),
        )}
      </div>
      <div className="flex gap-2">
        <Button type="submit" size="small" loading={saving}>
          Add row
        </Button>
        <Button type="button" size="small" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
