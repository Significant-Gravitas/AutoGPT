import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";

const AVATAR_SIZE = 36;

interface Props {
  task: SuggestedTask;
  owner: RecommendedExpert | null;
  isFirst: boolean;
}

// One task as the plan step shows it: read-only, with who takes it.
export function PlanTaskCard({ task, owner, isFirst }: Props) {
  return (
    <li
      className={cn(
        "grid grid-cols-[2.25rem_minmax(0,1fr)] items-start gap-3 rounded-2xl border bg-surface px-3.5 py-3 [@media(max-height:780px)]:py-2.5",
        isFirst
          ? "border-brand/30 ring-2 ring-accent-surface"
          : "border-border-base",
      )}
    >
      {owner ? (
        <ExpertAvatar
          name={owner.name}
          avatarUrl={owner.avatar_url ?? null}
          size={AVATAR_SIZE}
        />
      ) : (
        <AutopilotAvatar size={AVATAR_SIZE} />
      )}
      <div className="flex min-w-0 flex-col items-start">
        {isFirst ? (
          <span className="mb-1.5 rounded-full bg-accent-surface px-2 py-0.5 text-caption-sm font-semibold tracking-eyebrow text-accent-strong uppercase">
            Start here
          </span>
        ) : null}
        <Text variant="body-medium" as="span" className="text-pretty">
          {task.title}
        </Text>
        {task.why ? (
          <Text variant="small" as="span" tone="muted" className="mt-0.5">
            {task.why}
          </Text>
        ) : null}
        <OwnerLine owner={owner} />
      </div>
    </li>
  );
}

function OwnerLine({ owner }: { owner: RecommendedExpert | null }) {
  if (!owner)
    return (
      // Text has no accent tone, so the role classes go on the span.
      <span className="mt-1.5 font-sans text-caption text-accent-strong">
        Otto handles this one
      </span>
    );
  return (
    <Text variant="small" as="span" tone="secondary" className="mt-1.5">
      <span className="font-medium text-foreground-strong">{owner.name}</span>,{" "}
      {getExpertRoleLabel(owner.role)}{" "}
      <span className="text-foreground-subtle">· joins your team</span>
    </Text>
  );
}
