import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import NextLink from "next/link";
import {
  getMemberStatusLabel,
  getTeamSummary,
  type MemberStatus,
  type TeamMember,
  VISIBLE_MEMBERS,
} from "../helpers";

const STATUS_DOTS: Record<MemberStatus, string> = {
  // TODO(tokens): no exact role: bg-sky-500 #0EA5E9
  installing: "bg-sky-500 animate-pulse",
  unchecked: "bg-warning",
  failed: "bg-error",
  needs_connection: "bg-warning",
  needs_setup: "bg-warning",
  working: "bg-primary animate-pulse",
  // TODO(tokens): no exact role: bg-emerald-500 #10B981
  ready: "bg-emerald-500",
};

interface Props {
  team: TeamMember[];
}

export function TeamCard({ team }: Props) {
  const visible = team.slice(0, VISIBLE_MEMBERS);
  const hidden = team.length - visible.length;
  const isSettingUp = team.some((member) => member.status === "installing");

  return (
    <section
      aria-label="Your team"
      className="mt-5 overflow-hidden rounded-lg border border-border-base bg-surface md:-mx-5"
    >
      <div className="flex items-baseline justify-between gap-3 border-b border-separator px-4 py-2.5">
        <Text variant="body-medium" tone="primary" as="h3">
          Your team
        </Text>
        <Text variant="small" tone="muted" as="span">
          {getTeamSummary(team)}
        </Text>
      </div>
      <ul className="divide-y divide-separator">
        {visible.map((member) => (
          <li
            key={member.expert.id}
            className="flex min-w-0 items-center gap-3 px-4 py-3"
          >
            <ExpertAvatar
              name={member.expert.name}
              avatarUrl={member.expert.avatar_url ?? null}
              color={member.expert.color}
              size={32}
            />
            <div className="min-w-0 flex-1">
              <div className="flex min-w-0 items-baseline gap-2">
                <Text
                  variant="body-medium"
                  tone="primary"
                  as="span"
                  className="truncate"
                >
                  {member.expert.name}
                </Text>
                {/* A phone has room for the name and status only. */}
                <Text
                  variant="small"
                  tone="muted"
                  as="span"
                  className="truncate max-sm:hidden"
                >
                  {member.area}
                </Text>
              </div>
              {member.schedule && (
                <Text variant="small" tone="secondary" className="text-pretty">
                  {member.schedule}
                </Text>
              )}
            </div>
            <Text
              variant="small-medium"
              tone="muted"
              as="span"
              role="status"
              className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap"
            >
              <span
                aria-hidden
                className={cn(
                  "size-1.5 rounded-full",
                  STATUS_DOTS[member.status],
                )}
              />
              {getMemberStatusLabel(member.status)}
            </Text>
          </li>
        ))}
      </ul>
      {(hidden > 0 || isSettingUp) && (
        <div className="flex flex-wrap items-baseline justify-between gap-2 border-t border-separator px-4 py-2">
          {isSettingUp && (
            <Text variant="small" tone="muted" as="span">
              Each row turns Ready when its setup finishes.
            </Text>
          )}
          {hidden > 0 && (
            <NextLink
              href="/team"
              className="ml-auto text-sm font-medium text-foreground-secondary hover:text-foreground-strong"
            >
              and {hidden} more
            </NextLink>
          )}
        </div>
      )}
    </section>
  );
}
