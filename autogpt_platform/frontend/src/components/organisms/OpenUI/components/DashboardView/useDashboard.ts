import { useGetFieldValue, useStateField } from "@openuidev/react-lang";
import { z } from "zod/v4";
import { chartDataError, type Dashboard } from "@/lib/openui/catalog-dashboard";
import {
  calculatedRows,
  rowsSchema,
  type DataRow,
} from "@/lib/openui/data-values";
import { asNumber } from "@/lib/openui/calculations";

interface Filters {
  query: string;
  categories: Record<string, string>;
  ranges: Record<string, { min: string; max: string }>;
  selectedOnly: boolean;
}

export function useDashboard(props: z.infer<typeof Dashboard.props>) {
  const get = useGetFieldValue();
  const field = useStateField<Filters>(`dashboard:${props.name}`, {
    query: "",
    categories: {},
    ranges: {},
    selectedOnly: false,
  });
  const checked = z
    .object({
      query: z.string(),
      categories: z.record(z.string(), z.string()),
      ranges: z.record(
        z.string(),
        z.object({ min: z.string(), max: z.string() }),
      ),
      selectedOnly: z.boolean(),
    })
    .safeParse(field.value);
  const filters = checked.success
    ? checked.data
    : { query: "", categories: {}, ranges: {}, selectedOnly: false };
  const raw = props.binding
    ? get(props.binding.form, `${props.binding.field}_rows`)
    : props.rows;
  const parsed = rowsSchema.safeParse(raw);
  const rows = parsed.success ? calculatedRows(props.columns, parsed.data) : [];
  const selected = props.binding
    ? get(props.binding.form, `${props.binding.field}_selected`)
    : [];
  const sourceValid =
    !props.binding ||
    (parsed.success &&
      get(props.binding.form, `${props.binding.field}__valid`) !== false);
  const rangeError = Object.values(filters.ranges).some(
    ({ min, max }) =>
      (min !== "" && asNumber(min) === null) ||
      (max !== "" && asNumber(max) === null) ||
      (min !== "" && max !== "" && Number(min) > Number(max)),
  );
  const filtered = rows.filter((row) =>
    matches(row, filters, Array.isArray(selected) ? selected : []),
  );
  function category(key: string, value: string) {
    field.setValue({
      ...filters,
      categories: { ...filters.categories, [key]: value },
    });
  }
  function range(key: string, bound: "min" | "max", value: string) {
    field.setValue({
      ...filters,
      ranges: {
        ...filters.ranges,
        [key]: {
          ...(filters.ranges[key] ?? { min: "", max: "" }),
          [bound]: value,
        },
      },
    });
  }
  function reset() {
    field.setValue({
      query: "",
      categories: {},
      ranges: {},
      selectedOnly: false,
    });
  }
  return {
    rows,
    filtered,
    filters,
    category,
    range,
    reset,
    setFilters: field.setValue,
    error: !sourceValid
      ? "Correct the highlighted grid inputs to update this dashboard."
      : rangeError
        ? "Enter valid filter bounds, with minimum at or below maximum."
        : chartDataError(props.charts, props.columns, filtered),
  };
}

function matches(row: DataRow, filters: Filters, selected: string[]) {
  if (filters.selectedOnly && !selected.includes(row.id)) return false;
  if (
    !Object.values(row.cells).some((cell) =>
      String(cell ?? "")
        .toLowerCase()
        .includes(filters.query.toLowerCase()),
    )
  )
    return false;
  if (
    Object.entries(filters.categories).some(
      ([key, value]) => value !== "" && String(row.cells[key] ?? "") !== value,
    )
  )
    return false;
  return Object.entries(filters.ranges).every(([key, { min, max }]) => {
    const value = asNumber(row.cells[key]);
    if (min === "" && max === "") return true;
    return (
      value !== null &&
      (min === "" || value >= Number(min)) &&
      (max === "" || value <= Number(max))
    );
  });
}
