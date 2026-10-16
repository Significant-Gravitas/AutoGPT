import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

interface Props {
  expert: ExpertIdentity | undefined;
}

export function ExpertByline({ expert }: Props) {
  const name = expert?.name ?? AUTOPILOT_NAME;
  const jobTitle = expert ? expert.jobTitle || expert.role : "Head of AI";

  return (
    <div className="flex items-center gap-1.5 pr-6">
      {expert ? (
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatarUrl}
          color={expert.color}
          size={20}
          className="rounded-full border border-zinc-200"
        />
      ) : (
        <AutopilotAvatar
          size={20}
          className="rounded-full border border-zinc-200"
        />
      )}
      <Text variant="small" tone="muted" className="shrink-0">
        {name}
      </Text>
      {jobTitle && (
        <Text variant="small" className="truncate text-zinc-400">
          · {jobTitle}
        </Text>
      )}
    </div>
  );
}
