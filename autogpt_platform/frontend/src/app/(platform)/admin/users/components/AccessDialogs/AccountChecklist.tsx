"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { cn } from "@/lib/utils";
import { Alert02Icon, Tick01Icon } from "@hugeicons/core-free-icons";
import type { BanAccountResult } from "../../useBanAccounts";

export interface ChecklistAccount {
  id: string;
  email: string;
  name: string;
  banned: boolean;
  badges: string[];
}

interface Props {
  accounts: ChecklistAccount[];
  ticked: Set<string>;
  onToggle: (id: string) => void;
  results: Map<string, BanAccountResult>;
  maxHeightClassName: string;
}

export function AccountChecklist({
  accounts,
  ticked,
  onToggle,
  results,
  maxHeightClassName,
}: Props) {
  return (
    <ul
      className={cn(
        "divide-y divide-zinc-200 overflow-y-auto rounded-xl border border-zinc-200",
        maxHeightClassName,
      )}
    >
      {accounts.map((account) => {
        const result = results.get(account.id);
        return (
          <li key={account.id} className="flex items-center gap-3 px-3 py-2">
            <Checkbox
              id={`ban-${account.id}`}
              checked={!account.banned && ticked.has(account.id)}
              disabled={account.banned}
              onCheckedChange={() => onToggle(account.id)}
              aria-label={`Ban ${account.email}`}
            />
            <label
              htmlFor={`ban-${account.id}`}
              className={cn(
                "flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2",
                account.banned && "text-zinc-400 line-through",
              )}
            >
              <Text
                variant="body"
                as="span"
                className="truncate"
                translate="no"
              >
                {account.email}
              </Text>
              {account.name.trim() ? (
                <Text variant="small" as="span" className="text-zinc-500">
                  {account.name}
                </Text>
              ) : null}
            </label>
            {account.banned ? (
              <Text variant="small" as="span" className="text-zinc-500">
                already banned
              </Text>
            ) : (
              account.badges.map((badge) => (
                <Badge key={badge} variant="info" size="small">
                  {badge}
                </Badge>
              ))
            )}
            {result ? <ResultMark result={result} /> : null}
          </li>
        );
      })}
    </ul>
  );
}

function ResultMark({ result }: { result: BanAccountResult }) {
  if (result.ok) {
    return (
      <span className="flex items-center gap-1 text-emerald-700">
        <Icon icon={Tick01Icon} size={14} />
        <Text variant="small" as="span" className="text-emerald-700">
          banned
        </Text>
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 text-red-700">
      <Icon icon={Alert02Icon} size={14} />
      <Text variant="small" as="span" className="text-red-700">
        {result.error ?? "failed"}
      </Text>
    </span>
  );
}
