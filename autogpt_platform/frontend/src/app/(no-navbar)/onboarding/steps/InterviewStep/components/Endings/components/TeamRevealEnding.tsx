"use client";

import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

interface Props {
  team: RecommendedExpert[];
  isLeaving: boolean;
  onMeet: () => void;
}

// The team the hand-off proposed, once it has landed.
export function TeamRevealEnding({ team, isLeaving, onMeet }: Props) {
  return (
    <section
      aria-label="Your team"
      className="flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-5"
    >
      <Text variant="large-medium">Here&apos;s your team</Text>
      <ul className="flex flex-col gap-3">
        {team.map((expert) => (
          <li key={expert.template_id} className="flex items-center gap-3">
            <ExpertAvatar
              name={expert.name}
              avatarUrl={expert.avatar_url ?? null}
              size={40}
            />
            <div className="flex min-w-0 flex-col">
              <Text variant="body-medium">{expert.name}</Text>
              <Text variant="small" tone="muted">
                {expert.role}
              </Text>
            </div>
          </li>
        ))}
      </ul>
      <Button
        variant="primary"
        size="large"
        loading={isLeaving}
        disabled={isLeaving}
        onClick={onMeet}
      >
        Meet your team
      </Button>
    </section>
  );
}
