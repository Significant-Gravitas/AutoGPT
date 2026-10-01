"use client";

import { useState } from "react";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import type { ChatRule, RuleScope } from "../../helpers";

interface Props {
  label: string;
  subjectName: string;
  expertName: string | null;
  rules: ChatRule[];
  loading: boolean;
  disabled: boolean;
  onApprove: (rule?: ChatRule, scope?: RuleScope) => void;
}

export function ApproveSplitButton({
  label,
  subjectName,
  expertName,
  rules,
  loading,
  disabled,
  onApprove,
}: Props) {
  const [scope, setScope] = useState<RuleScope>("expert");
  const who = expertName ?? AUTOPILOT_NAME;
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
        <DropdownMenuContent align="start" className="w-80">
          {rules.map((rule) => {
            const [title, detail] = ruleCopy(rule, scope, subjectName, who);
            return (
              <DropdownMenuItem
                key={rule}
                onSelect={() => onApprove(rule, scope)}
                className="flex flex-col items-start gap-0.5 py-2"
              >
                <span className="text-sm text-zinc-900">{title}</span>
                <span className="text-xs text-zinc-500">{detail}</span>
              </DropdownMenuItem>
            );
          })}
          <DropdownMenuSeparator />
          <DropdownMenuLabel className="text-xs font-normal text-zinc-500">
            Applies to
          </DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={scope}
            onValueChange={(value) => setScope(value as RuleScope)}
          >
            {SCOPES.map((option) => (
              <DropdownMenuRadioItem
                key={option}
                value={option}
                // Picking a scope keeps the menu open so the action can follow.
                onSelect={(event) => event.preventDefault()}
                className="text-sm text-zinc-900"
              >
                {scopeLabel(option, who)}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

const SCOPES: RuleScope[] = ["chat", "expert", "team"];

function scopeLabel(scope: RuleScope, who: string) {
  if (scope === "chat") return "This chat";
  if (scope === "expert") return `${who}, every chat`;
  return "Every Expert on my team";
}

// The supervisor is always Otto, whichever Expert's chat the card is in.
function ruleCopy(
  rule: ChatRule,
  scope: RuleScope,
  subject: string,
  who: string,
): [string, string] {
  if (rule === "allow") {
    const detail = {
      chat: `${who} won't ask again for this in this chat`,
      expert: `${who} won't ask again for this in any chat`,
      team: "No Expert on your team will ask again for this",
    }[scope];
    return [`Approve ${subject} from now on`, detail];
  }
  const runs = {
    chat: "each time it runs in this chat",
    expert: `each time ${who} runs it, in any chat`,
    team: "each time any Expert on your team runs it",
  }[scope];
  return [
    `Let ${AUTOPILOT_NAME} judge ${subject} from now on`,
    `A check decides ${runs}, and asks you only when it isn't sure`,
  ];
}
