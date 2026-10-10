import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { CostTable } from "@/lib/openui/catalog-connected";
import { formatCalculated } from "@/lib/openui/calculations";
import { useOpenUIDisabled } from "../../interactionContext";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { CostRow } from "./CostRow";
import { costTotal, readCostRows, type CostRowDraft } from "./helpers";

export function CostTableView({
  props,
}: ComponentRenderProps<z.infer<typeof CostTable.props>>) {
  const disabled = useOpenUIDisabled();
  const field = useFieldView<CostRowDraft[]>(props.name, props.rows ?? []);
  const rows = readCostRows(field.value, props.rows ?? []);
  const total = costTotal(rows);
  useDerivedField(`${props.name}_total`, total);
  useDerivedField(`${props.name}_total__valid`, total !== null);
  return (
    <fieldset disabled={disabled} className="min-w-0">
      <legend className="text-sm font-semibold text-foreground">
        {props.title}
      </legend>
      <p className="mt-1 text-xs text-muted-foreground">
        Edit quantities and prices, or leave an item out. Prices in{" "}
        {props.currency}.
      </p>
      {rows.map((row) => (
        <CostRow
          key={row.id}
          row={row}
          currency={props.currency}
          disabled={disabled}
          onChange={(changed) =>
            field.setValue(
              rows.map((item) => (item.id === changed.id ? changed : item)),
            )
          }
        />
      ))}
      <div className="flex items-center justify-between gap-3 border-t border-border py-3 text-sm font-semibold text-foreground">
        <span>Subtotal</span>
        <output aria-live="polite">
          {total === null
            ? "Check your inputs"
            : formatCalculated(total, "currency", props.currency, 2)}
        </output>
      </div>
    </fieldset>
  );
}
