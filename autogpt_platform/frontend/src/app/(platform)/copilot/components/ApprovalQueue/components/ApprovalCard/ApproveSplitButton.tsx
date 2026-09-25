"use client";

import { useState } from "react";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Switch } from "@/components/atoms/Switch/Switch";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import type { ChatRule } from "../../helpers";

interface Props {
  label: string;
  subjectName: string;
  rules: ChatRule[];
  loading: boolean;
  disabled: boolean;
  onApprove: (rule?: ChatRule, team?: boolean) => void;
}

const TEAM_LABEL = "Apply to all Experts in my Team";

const RULE_COPY: Record<
  ChatRule,
  (subject: string, team: boolean) => [string, string]
> = {
  allow: (subject, team) =>
    team
      ? [
          "Approve for all my chats",
          `${subject} runs without asking in all your chats until you revoke it`,
        ]
      : [
          "Approve for this chat",
          `${subject} runs without asking until this chat ends`,
        ],
  judge: (_, team) => [
    `Let ${AUTOPILOT_NAME} judge from now on`,
    team
      ? "A check decides each time in all your chats and asks you only when it isn't sure"
      : "A check decides each time and asks you only when it isn't sure",
  ],
};

export function ApproveSplitButton({
  label,
  subjectName,
  rules,
  loading,
  disabled,
  onApprove,
}: Props) {
  const [team, setTeam] = useState(false);
  const main = (
    <Button
      size="small"
      variant="primary"
      loading={loading}
      disabled={disabled}
      onClick={() => onApprove()}
      className={
        rules.length > 0 ? "min-w-0 flex-1 rounded-r-none pr-2.5" : "min-w-0"
      }
    >
      {loading ? "Approving…" : label}
    </Button>
  );
  if (rules.length === 0) return main;

  return (
    <div className="inline-flex">
      {main}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            size="small"
            variant="primary"
            disabled={disabled}
            aria-label="More ways to approve"
            withTooltip={false}
            className="min-w-0 rounded-l-none border-l-zinc-600 px-2"
          >
            <Icon icon={ArrowDown01Icon} size={14} aria-hidden />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-72">
          {rules.map((rule) => {
            const [title, detail] = RULE_COPY[rule](subjectName, team);
            return (
              <DropdownMenuItem
                key={rule}
                onSelect={() => onApprove(rule, team)}
                className="flex flex-col items-start gap-0.5 py-2"
              >
                <span className="text-sm text-zinc-900">{title}</span>
                <span className="text-xs text-zinc-500">{detail}</span>
              </DropdownMenuItem>
            );
          })}
          <DropdownMenuSeparator />
          {/* Toggling keeps the menu open so the choice can follow. */}
          <DropdownMenuItem
            role="menuitemcheckbox"
            aria-checked={team}
            onSelect={(event) => {
              event.preventDefault();
              setTeam(!team);
            }}
            className="flex items-center justify-between gap-3 py-2"
          >
            <span className="text-sm text-zinc-900">{TEAM_LABEL}</span>
            <Switch
              checked={team}
              tabIndex={-1}
              aria-hidden
              className="pointer-events-none"
            />
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
