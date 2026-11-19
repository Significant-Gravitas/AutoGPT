"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";

export interface Choice {
  value: string;
  label: string;
}

interface Props {
  /** What it picks, for the accessible name: "Owner", "Start". */
  name: string;
  value: string;
  choices: Choice[];
  /** Set away from its default: drawn filled. */
  isActive: boolean;
  onChange: (value: string) => void;
  /** What the pill says, when not the choice's own label. */
  display?: string;
}

/** One compact choice: a pill showing it, a menu to change it. */
export function ChoiceMenu({
  name,
  value,
  choices,
  isActive,
  onChange,
  display,
}: Props) {
  const current = choices.find((choice) => choice.value === value);
  const label = display ?? current?.label ?? choices[0].label;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant={isActive ? "secondary" : "ghost-muted"}
          size="xs"
          shape="pill"
          aria-label={`${name}: ${label}. Change ${name.toLowerCase()}`}
          rightIcon={<Icon icon={ArrowDown01Icon} size={12} tone="faint" />}
        >
          {label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-72 overflow-y-auto">
        <DropdownMenuRadioGroup
          value={current?.value ?? choices[0].value}
          onValueChange={onChange}
        >
          {choices.map((choice) => (
            <DropdownMenuRadioItem key={choice.value} value={choice.value}>
              {choice.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
