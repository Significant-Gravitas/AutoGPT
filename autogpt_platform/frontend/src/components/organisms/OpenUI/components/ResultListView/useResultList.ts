import { useState } from "react";
import type { z } from "zod/v4";
import type { ResultList } from "@/lib/openui/catalog-results";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";

export function useResultList(props: z.infer<typeof ResultList.props>) {
  const field = useFieldView(props.name, props.value);
  const selected = Array.isArray(field.value)
    ? field.value.filter((id) =>
        props.options.some((option) => option.id === id),
      )
    : [];
  const options = props.options.filter((option) =>
    selected.includes(option.id),
  );
  const amount =
    options.length && options.every((option) => option.amount != null)
      ? options.reduce((sum, option) => sum + (option.amount ?? 0), 0)
      : null;
  useDerivedField(`${props.name}_amount`, amount);
  useDerivedField(`${props.name}_count`, selected.length);
  useDerivedField(`${props.name}__valid`, !props.required || !!selected.length);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [previous, setPrevious] = useState<string[] | null>(null);
  const matches = props.options.filter((option) =>
    `${option.title} ${option.description} ${option.facts.map((fact) => `${fact.label} ${fact.value}`).join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const currentPage = Math.min(
    page,
    Math.max(0, Math.ceil(matches.length / 6) - 1),
  );
  function choose(id: string) {
    setPrevious(selected);
    field.setValue(
      selected.includes(id)
        ? selected.filter((value) => value !== id)
        : props.selection === "single"
          ? [id]
          : [...selected, id],
    );
  }
  function undo() {
    if (previous) {
      field.setValue(previous);
      setPrevious(null);
    }
  }
  return {
    selected,
    query,
    setQuery,
    page: currentPage,
    setPage,
    matches,
    visible: matches.slice(currentPage * 6, currentPage * 6 + 6),
    choose,
    undo,
    canUndo: previous !== null,
  };
}
