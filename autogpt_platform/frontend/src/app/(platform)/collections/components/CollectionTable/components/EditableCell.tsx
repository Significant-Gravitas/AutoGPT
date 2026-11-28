import { Button } from "@/components/atoms/Button/Button";
import { Checkbox } from "@/components/atoms/Checkbox/Checkbox";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { useState } from "react";
import {
  cellText,
  editText,
  inputType,
  parseInput,
  type TableColumn,
  type TableRow,
} from "../helpers";

interface Props {
  column: TableColumn;
  row: TableRow;
  rowLabel: string;
  readOnly: boolean;
  onSave(value: unknown): Promise<boolean>;
  onAccept(): Promise<boolean>;
}

export function EditableCell({
  column,
  row,
  rowLabel,
  readOnly,
  onSave,
  onAccept,
}: Props) {
  const value = row.values[column.field];
  const label = `${column.field} of ${rowLabel}`;
  const pending = row.pending?.[column.field];
  const numeric = column.type === "number";
  return (
    <div className={cn("flex flex-col gap-1", numeric && "items-end")}>
      <CellValue
        column={column}
        value={value}
        label={label}
        readOnly={readOnly}
        onSave={onSave}
      />
      {pending !== undefined && pending !== null ? (
        <div
          className={cn(
            "flex max-w-56 flex-wrap items-center gap-x-1.5",
            numeric && "justify-end text-right",
          )}
        >
          <Text variant="small" tone="secondary">
            {writerName(row.written_by)} has a newer value:{" "}
            {cellText(column, pending)}
          </Text>
          {readOnly ? null : (
            <Button
              size="xs"
              variant="ghost"
              aria-label={`Use the newer value for ${label}`}
              onClick={() => onAccept()}
            >
              Use it
            </Button>
          )}
        </div>
      ) : null}
    </div>
  );
}

function CellValue({
  column,
  value,
  label,
  readOnly,
  onSave,
}: {
  column: TableColumn;
  value: unknown;
  label: string;
  readOnly: boolean;
  onSave(value: unknown): Promise<boolean>;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  if (column.type === "checkbox") {
    return (
      <Checkbox
        aria-label={label}
        checked={value === true}
        disabled={readOnly}
        onCheckedChange={(checked) => onSave(checked === true)}
      />
    );
  }
  if (column.type === "select" && !readOnly) {
    return (
      <Select
        id={`cell-${label}`}
        label={label}
        hideLabel
        size="small"
        wrapperClassName="!mb-0"
        value={typeof value === "string" ? value : ""}
        placeholder="–"
        options={(column.options ?? []).map((o) => ({ value: o, label: o }))}
        onValueChange={(next) => onSave(next)}
      />
    );
  }
  if (draft === null) {
    const text = cellText(column, value);
    const link =
      column.type === "url" && typeof value === "string" ? value : null;
    if (readOnly) {
      return link ? (
        <Link href={link} isExternal>
          {text}
        </Link>
      ) : (
        <Display column={column} text={text} />
      );
    }
    return (
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          aria-label={`Edit ${label}`}
          className={cn(
            "min-h-6 w-full cursor-text rounded-md hover:bg-surface-muted",
            column.type === "number" ? "text-right" : "text-left",
          )}
          onClick={() => setDraft(editText(value))}
        >
          <Display column={column} text={text} />
        </button>
        {link ? (
          <Link href={link} isExternal>
            Open
          </Link>
        ) : null}
      </div>
    );
  }

  async function commit(text: string) {
    if (await onSave(parseInput(column.type, text))) setDraft(null);
  }

  return (
    <Input
      id={`cell-${label}`}
      label={label}
      hideLabel
      size="small"
      wrapperClassName="!mb-0"
      autoFocus
      type={inputType(column.type)}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => commit(draft)}
      onKeyDown={(e) => {
        if (isKey(e, "Enter")) commit(draft);
        if (isKey(e, "Escape")) setDraft(null);
      }}
    />
  );
}

function Display({ column, text }: { column: TableColumn; text: string }) {
  if (!text) {
    return (
      <Text as="span" variant="body" tone="secondary">
        –
      </Text>
    );
  }
  return (
    <Text
      as="span"
      variant="body"
      tabular={column.type === "number"}
      className="break-words"
    >
      {text}
    </Text>
  );
}

function writerName(writtenBy: string) {
  if (writtenBy.startsWith("expert:")) return "Your expert";
  if (writtenBy === "otto") return "Otto";
  return "The workflow";
}
