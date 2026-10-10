import { useState } from "react";
import type { z } from "zod/v4";
import type { Comparison } from "@/lib/openui/catalog-connected";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";

export function useComparisonView(props: z.infer<typeof Comparison.props>) {
  const field = useFieldView(props.name, props.value ?? "");
  const [history, setHistory] = useState<string[]>([]);
  const options = props.options ?? [];
  const selected = options.find((option) => option.id === field.value);
  useDerivedField(`${props.name}_amount`, selected?.amount ?? null);
  useDerivedField(`${props.name}_label`, selected?.title ?? "");

  function choose(id: string) {
    if (id === field.value) return;
    setHistory((items) => [...items.slice(-19), String(field.value)]);
    field.setValue(id);
  }
  function undo() {
    if (!history.length) return;
    field.setValue(history[history.length - 1]);
    setHistory((items) => items.slice(0, -1));
  }
  return { selected, choose, undo, canUndo: history.length > 0 };
}
