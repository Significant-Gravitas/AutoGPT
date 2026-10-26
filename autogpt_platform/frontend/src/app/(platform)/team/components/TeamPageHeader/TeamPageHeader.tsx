"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { TeamHeaderActions } from "../TeamHeaderActions";

interface Props {
  experts: Expert[];
}

const AVATAR_CLASS =
  // TODO(tokens): no exact role: border-[#e3e3e3] #E3E3E3, ring-[#f9f9f9] #F9F9F9
  "rounded-full border border-[#e3e3e3] ring-2 ring-[#f9f9f9]";

// Same motion as the side panel's empty-state tiles: the outer two lean out
// and spread a little further on hover.
const POSITIONS = [
  "left-0 top-1 z-0 -rotate-6 group-hover:-translate-x-1 group-hover:-rotate-12",
  "right-0 top-1 z-0 rotate-6 group-hover:translate-x-1 group-hover:rotate-12",
];

/** Otto flanked by the first two hires, over the page title and a line on
 *  what the team is for. */
export function TeamPageHeader({ experts }: Props) {
  const flank = experts.slice(0, 2);

  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="group flex items-center gap-4">
        <div aria-hidden className="relative h-14 w-24 shrink-0">
          {flank.map((expert, index) => (
            <span
              key={expert.id}
              className={cn(
                "absolute transition-transform duration-300 ease-out motion-reduce:transition-none",
                POSITIONS[index],
              )}
            >
              <ExpertAvatar
                name={expert.name}
                avatarUrl={expert.avatar_url}
                color={expert.color}
                size={44}
                className={AVATAR_CLASS}
              />
            </span>
          ))}
          <span className="absolute left-1/2 top-0 z-10 -translate-x-1/2 transition-transform duration-300 ease-out group-hover:-translate-y-0.5 motion-reduce:transition-none">
            <AutopilotAvatar size={52} className={AVATAR_CLASS} />
          </span>
        </div>
        <div className="flex min-w-0 flex-col gap-0.5">
          <Text variant="h4" as="h1" className="!text-foreground-strong">
            Team
          </Text>
          <Text variant="body" className="!text-foreground-tertiary">
            Otto and your hired experts, ready to work.
          </Text>
        </div>
      </div>
      <TeamHeaderActions />
    </div>
  );
}
