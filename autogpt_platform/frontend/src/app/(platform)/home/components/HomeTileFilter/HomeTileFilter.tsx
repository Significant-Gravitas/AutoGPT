"use client";

import { FilterHorizontalIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";

interface Props {
  ariaLabelPrefix: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

export function HomeTileFilter({
  ariaLabelPrefix,
  value,
  options,
  onChange,
}: Props) {
  const activeLabel =
    options.find((option) => option.value === value)?.label ?? value;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="secondary"
          size="small"
          className="min-w-0"
          leftIcon={
            <Icon icon={FilterHorizontalIcon} size={15} aria-hidden="true" />
          }
          aria-label={`${ariaLabelPrefix}: ${activeLabel}`}
          unmask={false}
        >
          {activeLabel}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-36">
        <DropdownMenuRadioGroup value={value} onValueChange={onChange}>
          {options.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
