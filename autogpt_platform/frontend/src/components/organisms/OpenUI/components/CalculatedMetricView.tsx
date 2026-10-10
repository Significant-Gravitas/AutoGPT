import {
  useGetFieldValue,
  type ComponentRenderProps,
} from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { CalculatedMetric } from "@/lib/openui/catalog-connected";
import { formatCalculated } from "@/lib/openui/calculations";
import { parseFormula, evaluateFormula } from "@/lib/openui/formula";

export function CalculatedMetricView({
  props,
}: ComponentRenderProps<z.infer<typeof CalculatedMetric.props>>) {
  const get = useGetFieldValue();
  const { value, error } = metricResult(props, get);
  return (
    <section aria-label={props.label} className="space-y-1 py-2">
      <h3 className="text-sm text-muted-foreground">{props.label}</h3>
      <output
        aria-live="polite"
        className="block text-2xl font-semibold tabular-nums text-foreground"
      >
        {value === null
          ? "—"
          : formatCalculated(value, props.format, props.unit, props.precision)}
      </output>
      {value === null && (
        <p className="text-xs text-muted-foreground">{error}</p>
      )}
    </section>
  );
}

function metricResult(
  props: z.infer<typeof CalculatedMetric.props>,
  get: (form: string, name: string) => unknown,
) {
  try {
    const { root, references } = parseFormula(props.formula ?? "");
    if (
      references.some(({ name }) => get(props.form, `${name}__valid`) === false)
    )
      return {
        value: null,
        error: "Check the highlighted inputs to calculate this total.",
      };
    return {
      value: evaluateFormula(root, (name) => get(props.form, name)),
      error: "Complete your choices to calculate this total.",
    };
  } catch (error) {
    return {
      value: null,
      error: error instanceof Error ? error.message : "Check the calculation.",
    };
  }
}
