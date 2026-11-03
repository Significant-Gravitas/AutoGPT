import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import {
  EMPTY_FILTERS,
  type FilterChip,
  type PageFilters,
} from "../../helpers";

interface Props {
  chips: FilterChip[];
  onChange: (filters: PageFilters) => void;
}

const CHIP_CLASS =
  "inline-flex animate-inbox-enter items-center gap-1 rounded-full bg-surface-tray py-1 pr-1.5 pl-2.5 transition-[background-color,transform] duration-150 ease-out-expo hover:bg-surface-hover focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none active:scale-95";

export function ActiveFilters({ chips, onChange }: Props) {
  if (chips.length === 0) return null;

  return (
    <ul
      className="flex flex-wrap items-center gap-1.5"
      aria-label="Active filters"
    >
      {chips.map((chip) => (
        <li key={chip.id}>
          <button
            type="button"
            className={CHIP_CLASS}
            aria-label={`Remove filter ${chip.label}`}
            onClick={() => onChange(chip.next)}
          >
            <Text variant="small" tone="text" as="span">
              {chip.label}
            </Text>
            <Icon icon={Cancel01Icon} size={12} tone="quiet" />
          </button>
        </li>
      ))}
      <li>
        <button
          type="button"
          className="rounded-full px-2 py-1 transition-colors duration-150 ease-out-expo hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none"
          onClick={() => onChange(EMPTY_FILTERS)}
        >
          <Text variant="small" tone="quiet" as="span">
            Clear all
          </Text>
        </button>
      </li>
    </ul>
  );
}
