interface Props {
  isExpanded: boolean;
  total: number;
  /** Plural noun for the label, e.g. "workflows". */
  noun: string;
  onToggle: () => void;
}

export function ShelfMoreButton({ isExpanded, total, noun, onToggle }: Props) {
  return (
    <button
      type="button"
      aria-expanded={isExpanded}
      onClick={onToggle}
      className="mt-4 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
    >
      {isExpanded ? "Show fewer" : `Load all ${total.toLocaleString()} ${noun}`}
    </button>
  );
}
