import { formatCalculated } from "@/lib/openui/calculations";
import { CostInput } from "./CostInput";
import { rowTotal, type CostRowDraft } from "./helpers";

interface Props {
  row: CostRowDraft;
  currency: string;
  disabled: boolean;
  onChange: (row: CostRowDraft) => void;
}

export function CostRow({ row, currency, disabled, onChange }: Props) {
  const total = rowTotal(row);
  return (
    <div className="space-y-2 border-b border-border py-3 last:border-0">
      <div className="flex items-center justify-between gap-3">
        <label className="flex min-h-11 min-w-0 items-center gap-2 text-sm font-medium text-foreground">
          <input
            type="checkbox"
            className="h-4 w-4 shrink-0 accent-primary"
            checked={row.included}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...row, included: event.target.checked })
            }
            aria-label={`Include ${row.label}`}
          />
          <span className="break-words">{row.label}</span>
        </label>
        <output
          className="shrink-0 text-sm tabular-nums text-foreground"
          aria-label={`${row.label} total`}
        >
          {total === null
            ? "—"
            : formatCalculated(total, "currency", currency, 2)}
        </output>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <CostInput
          label={`${row.label} quantity`}
          kind="quantity"
          value={row.quantity}
          disabled={disabled || !row.included}
          onChange={(quantity) => onChange({ ...row, quantity })}
        />
        <CostInput
          label={`${row.label} unit price`}
          kind="unitPrice"
          value={row.unitPrice}
          disabled={disabled || !row.included}
          onChange={(unitPrice) => onChange({ ...row, unitPrice })}
        />
      </div>
    </div>
  );
}
