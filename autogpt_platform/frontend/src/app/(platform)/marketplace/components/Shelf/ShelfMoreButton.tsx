import { Button } from "@/components/atoms/Button/Button";

interface Props {
  isExpanded: boolean;
  /** How many tiles the button reveals. */
  count: number;
  /** False when the page cap keeps `count` short of the whole catalogue, so
   *  the label stops promising "all" — the header link covers the rest. */
  isAll?: boolean;
  /** Plural noun for the label, e.g. "workflows". */
  noun: string;
  onToggle: () => void;
}

export function ShelfMoreButton({
  isExpanded,
  count,
  isAll = true,
  noun,
  onToggle,
}: Props) {
  return (
    <Button
      size="small"
      aria-expanded={isExpanded}
      onClick={onToggle}
      unmask={false}
    >
      {isExpanded
        ? "Show fewer"
        : `Load ${isAll ? "all " : ""}${count.toLocaleString()} ${noun}`}
    </Button>
  );
}
