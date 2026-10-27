import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { type ConsentMoment, consentLine } from "./helpers";

const FACE_SIZE = 24;

interface Props {
  team: RecommendedExpert[];
  moment: ConsentMoment;
}

// Who joins, and when: the line that makes the plan a promise, not a hire.
export function PlanConsentLine({ team, moment }: Props) {
  return (
    <div className="flex items-center gap-2.5">
      {team.length > 0 ? (
        <span aria-hidden className="flex shrink-0 -space-x-1.5">
          {team.map((expert) => (
            <ExpertAvatar
              key={expert.template_id}
              name={expert.name}
              avatarUrl={expert.avatar_url ?? null}
              size={FACE_SIZE}
              className="rounded-full ring-2 ring-surface-muted"
            />
          ))}
        </span>
      ) : null}
      <Text variant="small" as="p" tone="secondary" className="text-pretty">
        {consentLine(
          team.map((expert) => expert.name),
          moment,
        )}
      </Text>
    </div>
  );
}
