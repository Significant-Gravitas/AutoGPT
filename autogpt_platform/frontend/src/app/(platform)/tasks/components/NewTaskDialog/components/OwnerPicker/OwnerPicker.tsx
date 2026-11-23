"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import type { OwnerChoice } from "../../helpers";
import { AddExpertItem } from "./AddExpertItem";

interface Props {
  value: OwnerChoice;
  choices: OwnerChoice[];
  onChange: (value: string) => void;
}

/** Who does the task: Otto, or an expert on the team. With no expert yet,
 *  it says where to add one. */
export function OwnerPicker({ value, choices, onChange }: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          shape="pill"
          aria-label={`Owner: ${value.label}. Change owner`}
          leftIcon={<ChoiceAvatar choice={value} />}
          rightIcon={<Icon icon={ArrowDown01Icon} tone="faint" />}
        >
          {value.label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="max-h-72 max-w-72 overflow-y-auto"
      >
        <DropdownMenuRadioGroup value={value.value} onValueChange={onChange}>
          {choices.map((choice) => (
            <DropdownMenuRadioItem key={choice.value} value={choice.value}>
              <span className="flex items-center gap-2">
                <ChoiceAvatar choice={choice} />
                {choice.label}
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
        {choices.length === 1 ? (
          <>
            <DropdownMenuSeparator />
            <AddExpertItem />
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ChoiceAvatar({ choice }: { choice: OwnerChoice }) {
  return (
    <span aria-hidden className="inline-flex">
      <ExpertAvatar
        name={choice.label}
        avatarUrl={choice.avatarUrl}
        size={18}
      />
    </span>
  );
}
