import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { FilterHorizontalIcon } from "@hugeicons/core-free-icons";
import { PRIORITY_LABELS } from "../../data/helpers";
import type { ExpertOption } from "../../data/types";
import {
  ALL_PRIORITIES,
  DUE_LABELS,
  expertChoices,
  toggleValue,
  type DueFilter,
  type PageFilters,
} from "../../helpers";

interface Props {
  filters: PageFilters;
  onChange: (filters: PageFilters) => void;
  experts: ExpertOption[];
}

const DUE_OPTIONS = Object.keys(DUE_LABELS) as DueFilter[];

function keepOpen(e: Event) {
  e.preventDefault();
}

export function FilterMenu({ filters, onChange, experts }: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="toggle"
          size="icon-xs"
          aria-label="Filter tasks"
          withTooltip={false}
        >
          <Icon icon={FilterHorizontalIcon} size={14} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuCheckboxItem
          checked={filters.mine}
          onSelect={keepOpen}
          onCheckedChange={(checked) =>
            onChange({ ...filters, mine: checked === true })
          }
        >
          Mine
        </DropdownMenuCheckboxItem>
        <DropdownMenuSeparator />
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Expert</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {expertChoices(experts).map((choice) => (
              <DropdownMenuCheckboxItem
                key={choice.id}
                checked={filters.assigneeIDs.includes(choice.id)}
                onSelect={keepOpen}
                onCheckedChange={() =>
                  onChange({
                    ...filters,
                    assigneeIDs: toggleValue(filters.assigneeIDs, choice.id),
                  })
                }
              >
                {choice.name}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Priority</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {ALL_PRIORITIES.map((priority) => (
              <DropdownMenuCheckboxItem
                key={priority}
                checked={filters.priority.includes(priority)}
                onSelect={keepOpen}
                onCheckedChange={() =>
                  onChange({
                    ...filters,
                    priority: toggleValue(filters.priority, priority),
                  })
                }
              >
                {PRIORITY_LABELS[priority]}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Due</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {DUE_OPTIONS.map((due) => (
              <DropdownMenuCheckboxItem
                key={due}
                checked={filters.due === due}
                onCheckedChange={(checked) =>
                  onChange({ ...filters, due: checked ? due : null })
                }
              >
                {DUE_LABELS[due]}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
