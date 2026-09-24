"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
  onApprove: (rule?: ChatRule) => void;
}

const RULE_COPY: Record<ChatRule, (subject: string) => [string, string]> = {
  allow: (subject) => [
    "Approve for this chat",
    `${subject} runs without asking until this chat ends`,
  ],
  judge: () => [
    `Let ${AUTOPILOT_NAME} judge from now on`,
    "A check decides each time and asks you only when it isn't sure",
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
            const [title, detail] = RULE_COPY[rule](subjectName);
            return (
              <DropdownMenuItem
                key={rule}
                onSelect={() => onApprove(rule)}
                className="flex flex-col items-start gap-0.5 py-2"
              >
                <span className="text-sm text-zinc-900">{title}</span>
                <span className="text-xs text-zinc-500">{detail}</span>
              </DropdownMenuItem>
            );
          })}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
