"use client";

import {
  Alert01Icon,
  ArrowDown01Icon,
  Calendar03Icon,
  PauseIcon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import type { TeamFilter } from "../../helpers";

const OPTIONS = [
  { value: "all", label: "All experts", icon: UserGroupIcon },
  { value: "scheduled", label: "Scheduled", icon: Calendar03Icon },
  { value: "needs-setup", label: "Needs setup", icon: Alert01Icon },
  { value: "paused", label: "Paused", icon: PauseIcon },
] as const;

interface Props {
  value: TeamFilter;
  onChange: (next: TeamFilter) => void;
}

/** A pill like the header's buttons, naming the current filter. Any choice
 *  other than "All experts" fills it in so a narrowed roster is obvious. */
export function TeamFilterMenu({ value, onChange }: Props) {
  const current =
    OPTIONS.find((option) => option.value === value) ?? OPTIONS[0];
  const isActive = value !== "all";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="secondary"
          size="small"
          aria-label="Filter experts"
          leadingIcon={current.icon}
          rightIcon={
            <Icon
              icon={ArrowDown01Icon}
              size={14}
              className="ml-0.5 text-zinc-400 transition-transform duration-200 group-data-[state=open]:rotate-180"
            />
          }
          className={cn(
            "group gap-1.5 font-medium",
            isActive &&
              "border-zinc-900 bg-zinc-900 text-white hover:border-zinc-800 hover:bg-zinc-800 [&_svg]:text-white",
          )}
        >
          {current.label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[12rem]">
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(next) => onChange(next as TeamFilter)}
        >
          {OPTIONS.map((option) => (
            <DropdownMenuRadioItem
              key={option.value}
              value={option.value}
              className="gap-2"
            >
              <Icon icon={option.icon} size={15} className="text-zinc-500" />
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
