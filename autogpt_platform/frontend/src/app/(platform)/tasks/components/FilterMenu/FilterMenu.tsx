import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { FilterHorizontalIcon } from "@hugeicons/core-free-icons";
import { PRIORITY_LABELS, STATUS_LABELS } from "../../data/helpers";
import type { ExpertOption } from "../../data/types";
import {
  ALL_PRIORITIES,
  ALL_STATUSES,
  DUE_LABELS,
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
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Expert</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {experts.length === 0 ? (
              <DropdownMenuCheckboxItem disabled checked={false}>
                No experts yet
              </DropdownMenuCheckboxItem>
            ) : (
              experts.map((expert) => (
                <DropdownMenuCheckboxItem
                  key={expert.id}
                  checked={filters.assigneeIDs.includes(expert.id)}
                  onSelect={keepOpen}
                  onCheckedChange={() =>
                    onChange({
                      ...filters,
                      assigneeIDs: toggleValue(filters.assigneeIDs, expert.id),
                    })
                  }
                >
                  {expert.name}
                </DropdownMenuCheckboxItem>
              ))
            )}
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
          <DropdownMenuSubTrigger>Status</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {ALL_STATUSES.map((status) => (
              <DropdownMenuCheckboxItem
                key={status}
                checked={filters.status.includes(status)}
                onSelect={keepOpen}
                onCheckedChange={() =>
                  onChange({
                    ...filters,
                    status: toggleValue(filters.status, status),
                  })
                }
              >
                {STATUS_LABELS[status]}
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
