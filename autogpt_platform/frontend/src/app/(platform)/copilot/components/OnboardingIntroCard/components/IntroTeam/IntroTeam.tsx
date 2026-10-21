"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import {
  Alert01Icon,
  BubbleChatIcon,
  Loading03Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import {
  getKickoffHref,
  getMemberArea,
  getMemberStatus,
  getMemberStatusLabel,
  type MemberStatus,
} from "./helpers";
import { useIntroTeam } from "./useIntroTeam";

const STATUS_ICONS = {
  installing: { icon: Loading03Icon, className: "animate-spin text-zinc-400" },
  ready: { icon: Tick02Icon, className: "text-emerald-600" },
  failed: { icon: Alert01Icon, className: "text-amber-600" },
} satisfies Record<MemberStatus, { icon: IconSvgElement; className: string }>;

/** The experts hired during onboarding, on the first screen after it: the
 *  greeting never names them, so without this they are only in the sidebar. */
export function IntroTeam() {
  const { experts, lead, trackChatOpened } = useIntroTeam();

  if (experts.length === 0) return null;

  return (
    <section aria-label="Your team" className="mt-5">
      <Text variant="body" tone="secondary" className="text-pretty">
        {lead}
      </Text>
      <ul className="-mx-5 mt-3 divide-y divide-zinc-100 overflow-hidden rounded-lg border border-zinc-200 bg-white">
        {experts.map((expert) => {
          const status = getMemberStatus(expert);
          const statusIcon = STATUS_ICONS[status];
          return (
            <li
              key={expert.id}
              className="flex min-w-0 items-center gap-3 px-4 py-2.5"
            >
              <ExpertAvatar
                name={expert.name}
                avatarUrl={expert.avatar_url ?? null}
                color={expert.color}
                size={32}
              />
              <div className="min-w-0 flex-1">
                <Text
                  variant="body-medium"
                  tone="primary"
                  className="truncate leading-5"
                >
                  {expert.name}
                </Text>
                <Text
                  variant="small"
                  tone="muted"
                  className="flex min-w-0 items-center gap-1.5"
                >
                  {/* A phone has room for the status only. */}
                  <span className="truncate max-sm:hidden">
                    {getMemberArea(expert)}
                  </span>
                  <span aria-hidden className="max-sm:hidden">
                    ·
                  </span>
                  <Icon
                    icon={statusIcon.icon}
                    size={12}
                    className={cn("shrink-0", statusIcon.className)}
                    aria-hidden
                  />
                  <span role="status" className="shrink-0 whitespace-nowrap">
                    {getMemberStatusLabel(status)}
                  </span>
                </Text>
              </div>
              <Button
                as="NextLink"
                href={getKickoffHref(expert.id)}
                variant="secondary"
                size="small"
                leadingIcon={BubbleChatIcon}
                aria-label={`Chat with ${expert.name}`}
                onClick={() => trackChatOpened(expert)}
                className="shrink-0"
              >
                Chat
              </Button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
