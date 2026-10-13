import { useId } from "react";
import { useState } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { SearchSelect } from "@/lib/openui/catalog-inputs";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { useOpenUIDisabled } from "../../interactionContext";
import { ControlValidity } from "../ControlValidity";

export function SearchSelectView({
  props,
}: ComponentRenderProps<z.infer<typeof SearchSelect.props>>) {
  const id = useId();
  const field = useFieldView(props.name, props.value);
  const disabled = useOpenUIDisabled();
  const [query, setQuery] = useState("");
  const selected = Array.isArray(field.value)
    ? field.value.filter((id) =>
        props.options.some((option) => option.id === id),
      )
    : [];
  const error =
    props.required && !selected.length
      ? "Choose an option before continuing."
      : "";
  useDerivedField(`${props.name}__valid`, !error);
  function choose(id: string) {
    field.setValue(
      selected.includes(id)
        ? selected.filter((value) => value !== id)
        : props.multiple
          ? [...selected, id]
          : [id],
    );
  }
  const options = props.options.filter((option) =>
    `${option.label} ${option.detail}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <fieldset disabled={disabled} className="space-y-2">
      <legend className="text-sm font-semibold">{props.label}</legend>
      <Input
        id={id}
        label={`Search ${props.label}`}
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div className="max-h-64 space-y-1 overflow-y-auto rounded-lg border border-border p-2">
        {options.map((option) => (
          <label
            key={option.id}
            className="flex min-h-11 cursor-pointer items-center gap-3 rounded-md p-2 hover:bg-muted"
          >
            <input
              type="checkbox"
              checked={selected.includes(option.id)}
              onChange={() => choose(option.id)}
            />
            <span className="min-w-0 text-sm">
              {option.label}
              <span className="block text-xs text-muted-foreground">
                {option.detail}
              </span>
            </span>
          </label>
        ))}
        {!options.length && <p className="text-sm">No matching options.</p>}
      </div>
      <p className="text-xs text-muted-foreground" aria-live="polite">
        {selected.length
          ? `Selected: ${props.options
              .filter((option) => selected.includes(option.id))
              .map((option) => option.label)
              .join(", ")}`
          : "Nothing selected"}
      </p>
      <Button
        type="button"
        variant="ghost"
        size="small"
        disabled={!selected.length}
        onClick={() => field.setValue([])}
      >
        Clear selection
      </Button>
      <ControlValidity label={props.label} error={error} />
    </fieldset>
  );
}
