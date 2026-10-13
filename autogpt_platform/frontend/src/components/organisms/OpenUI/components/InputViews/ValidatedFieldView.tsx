import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { ValidatedField } from "@/lib/openui/catalog-inputs";
import { cellError, typedCell } from "@/lib/openui/data-values";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { CellEditor } from "../CellEditor";
import { useOpenUIDisabled } from "../../interactionContext";

export function ValidatedFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof ValidatedField.props>>) {
  const field = useFieldView(`${props.name}__draft`, props.value ?? null);
  const disabled = useOpenUIDisabled();
  const error = cellError(field.value, props.column);
  useDerivedField(
    props.name,
    error ? null : typedCell(field.value, props.column),
  );
  useDerivedField(`${props.name}__valid`, !error);
  return (
    <CellEditor
      column={props.column}
      value={field.value}
      label={props.column.label}
      disabled={disabled}
      onChange={field.setValue}
    />
  );
}
