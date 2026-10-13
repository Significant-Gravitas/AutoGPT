import { useStateField } from "@openuidev/react-lang";
import { useOpenUIDisabled } from "../interactionContext";

export function useChecklistView(
  items: { done?: boolean }[],
  stateName: string,
) {
  const initialCompleted = items.flatMap((item, index) =>
    item.done === true ? [index] : [],
  );
  const field = useStateField<number[]>(stateName, initialCompleted);
  const completed = new Set<number>(
    Array.isArray(field.value)
      ? field.value.filter(
          (value): value is number =>
            typeof value === "number" &&
            Number.isInteger(value) &&
            value >= 0 &&
            value < items.length,
        )
      : initialCompleted,
  );
  const disabled = useOpenUIDisabled();

  function toggle(index: number) {
    if (disabled) return;
    const next = new Set(completed);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    field.setValue([...next]);
  }

  return { completed, disabled, toggle };
}
