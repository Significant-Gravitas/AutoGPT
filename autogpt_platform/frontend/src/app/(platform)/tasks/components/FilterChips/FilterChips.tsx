import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
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

export function FilterChips({ chips, onChange }: Props) {
  if (chips.length === 0) return null;

  return (
    <div
      className="flex flex-wrap items-center gap-2"
      aria-label="Active filters"
    >
      {chips.map((chip) => (
        <Button
          key={chip.id}
          variant="outline"
          size="xs"
          rightIcon={<Icon icon={Cancel01Icon} size={12} />}
          aria-label={`Remove filter ${chip.label}`}
          onClick={() => onChange(chip.next)}
        >
          {chip.label}
        </Button>
      ))}
      <Button
        variant="ghost-muted"
        size="xs"
        onClick={() => onChange(EMPTY_FILTERS)}
      >
        Clear all
      </Button>
    </div>
  );
}
