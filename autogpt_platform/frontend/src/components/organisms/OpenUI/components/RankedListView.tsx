import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { RankedList } from "@/lib/openui/catalog-results";
import { Button } from "@/components/atoms/Button/Button";
import { useFieldView } from "./useFieldView";
import { useOpenUIDisabled } from "../interactionContext";

export function RankedListView({
  props,
}: ComponentRenderProps<z.infer<typeof RankedList.props>>) {
  const field = useFieldView(props.name, props.value);
  const disabled = useOpenUIDisabled();
  const order =
    Array.isArray(field.value) &&
    field.value.length === props.items.length &&
    new Set(field.value).size === props.items.length &&
    field.value.every((id) => props.items.some((item) => item.id === id))
      ? field.value
      : props.value;
  function move(index: number, direction: number) {
    const next = [...order];
    [next[index], next[index + direction]] = [
      next[index + direction],
      next[index],
    ];
    field.setValue(next);
  }
  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-sm font-semibold">{props.label}</legend>
      <ol aria-live="polite" className="space-y-3">
        {order.map((id, index) => {
          const item = props.items.find((item) => item.id === id);
          return (
            <li
              key={id}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-border p-3"
            >
              <span className="text-sm font-semibold">
                {index + 1}. {item?.label}
              </span>
              <p className="min-w-0 flex-1 text-sm text-muted-foreground">
                {item?.detail}
              </p>
              <Button
                type="button"
                variant="ghost"
                size="small"
                disabled={!index}
                onClick={() => move(index, -1)}
                aria-label={`Move ${item?.label} up`}
              >
                ↑
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="small"
                disabled={index === order.length - 1}
                onClick={() => move(index, 1)}
                aria-label={`Move ${item?.label} down`}
              >
                ↓
              </Button>
            </li>
          );
        })}
      </ol>
    </fieldset>
  );
}
