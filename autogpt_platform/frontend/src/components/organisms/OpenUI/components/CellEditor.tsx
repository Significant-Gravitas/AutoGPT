import { useId } from "react";
import { Input } from "@/components/atoms/Input/Input";
import {
  cellError,
  type CellValue,
  type DataColumn,
} from "@/lib/openui/data-values";
import { useFieldValidation } from "./useFieldValidation";
import { FieldFeedback } from "./FieldFeedback";

interface Props {
  column: DataColumn;
  value: CellValue;
  label: string;
  disabled?: boolean;
  onChange: (value: CellValue) => void;
}

export function CellEditor({
  column,
  value,
  label,
  disabled,
  onChange,
}: Props) {
  const id = useId();
  const feedback = useFieldValidation(cellError(value, column));
  const { error, ...handlers } = feedback;
  if (column.type === "boolean")
    return (
      <label className="grid gap-1 text-xs">
        {label}
        <select
          {...handlers}
          required={column.required}
          value={value === null ? "" : String(value)}
          disabled={disabled}
          onChange={(event) =>
            onChange(
              event.target.value === "" ? null : event.target.value === "true",
            )
          }
          className="min-h-11 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">Unknown</option>
          <option value="true">Yes</option>
          <option value="false">No</option>
        </select>
        {error && <span className="text-destructive">{error}</span>}
      </label>
    );
  if (column.type === "select")
    return (
      <label className="grid gap-1 text-xs">
        {label}
        <select
          {...handlers}
          disabled={disabled}
          required={column.required}
          value={String(value ?? "")}
          onChange={(event) => onChange(event.target.value)}
          className="min-h-11 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">Choose…</option>
          {column.options?.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </select>
        {error && <span className="text-destructive">{error}</span>}
      </label>
    );
  return (
    <FieldFeedback id={`${id}-error`} error={error}>
      <Input
        {...handlers}
        id={id}
        label={label}
        value={String(value ?? "")}
        type={column.type === "date" ? "date" : "text"}
        inputMode={column.type === "number" ? "decimal" : undefined}
        maxLength={2000}
        required={column.required}
        disabled={disabled}
        aria-invalid={!!error}
        aria-describedby={`${id}-error`}
        onChange={(event) => onChange(event.target.value)}
      />
    </FieldFeedback>
  );
}
