"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { OTTO_VALUE, type AssigneeChoice } from "../../helpers";

interface Props {
  value: AssigneeChoice;
  choices: AssigneeChoice[];
  onChange: (value: string) => void;
}

export function AssigneeChip({ value, choices, onChange }: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          shape="pill"
          aria-label={`Assigned to ${value.label}. Change assignee`}
          leftIcon={<ChoiceAvatar choice={value} />}
          rightIcon={<Icon icon={ArrowDown01Icon} tone="faint" />}
        >
          {value.label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {choices.map((choice) => (
          <DropdownMenuItem
            key={choice.value}
            onSelect={() => onChange(choice.value)}
          >
            <span className="flex w-full items-center gap-2">
              <ChoiceAvatar choice={choice} />
              <span className="flex-1">{choice.label}</span>
              {choice.busy ? (
                <Text as="span" variant="small" tone="faint">
                  Busy
                </Text>
              ) : null}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ChoiceAvatar({ choice }: { choice: AssigneeChoice }) {
  if (choice.value === OTTO_VALUE) return <AutopilotAvatar size={18} />;
  return (
    <ExpertAvatar name={choice.label} avatarUrl={choice.avatarUrl} size={18} />
  );
}
