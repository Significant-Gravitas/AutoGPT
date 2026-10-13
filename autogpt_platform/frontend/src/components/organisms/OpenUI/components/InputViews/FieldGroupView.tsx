import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { FieldGroup } from "@/lib/openui/catalog-inputs";
import { Button } from "@/components/atoms/Button/Button";
import { CellEditor } from "../CellEditor";
import { ControlValidity } from "../ControlValidity";
import { useOpenUIDisabled } from "../../interactionContext";
import { useFieldGroup } from "./useFieldGroup";

export function FieldGroupView({
  props,
}: ComponentRenderProps<z.infer<typeof FieldGroup.props>>) {
  const group = useFieldGroup(props);
  const disabled = useOpenUIDisabled();
  if (!group.visible) return null;
  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-sm font-semibold">{props.label}</legend>
      {group.rows.map((row, index) => (
        <div
          key={row.id}
          className="space-y-3 rounded-lg border border-border p-3"
        >
          <p className="text-xs text-muted-foreground">
            {props.label} {index + 1}
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {props.columns.map((column) => (
              <CellEditor
                key={column.key}
                column={column}
                label={`${column.label} · ${row.id}`}
                value={row.cells[column.key] ?? null}
                disabled={disabled}
                onChange={(value) => group.edit(row.id, column.key, value)}
              />
            ))}
          </div>
          {props.repeatable && (
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={() => group.remove(row.id)}
            >
              Remove entry {index + 1}
            </Button>
          )}
        </div>
      ))}
      {props.repeatable && (
        <Button
          type="button"
          variant="secondary"
          size="small"
          disabled={group.rows.length >= 20}
          onClick={group.add}
        >
          Add {props.label.toLowerCase()}
        </Button>
      )}
      <ControlValidity label={props.label} error={group.error} />
    </fieldset>
  );
}
