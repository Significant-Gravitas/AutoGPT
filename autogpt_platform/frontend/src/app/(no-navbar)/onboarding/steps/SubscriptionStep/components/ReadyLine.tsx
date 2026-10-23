import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatReadyLine } from "./TrialWelcome/helpers";
import type { ReadyExpert } from "./useReadyExperts";

const AVATAR_SIZE = 24;
const MAX_AVATARS = 4;

interface Props {
  experts: ReadyExpert[];
}

// The welcome comes before the billing: the user's own setup, by name.
export function ReadyLine({ experts }: Props) {
  return (
    <p className="inline-flex max-w-full items-center gap-2 rounded-2xl bg-[#B6A4C8]/[0.24] py-1 pl-1 pr-3 font-sans text-xs font-medium text-zinc-800 sm:rounded-full">
      <span aria-hidden className="flex shrink-0 -space-x-1">
        <AutopilotAvatar size={AVATAR_SIZE} className="rounded-full" />
        {experts.slice(0, MAX_AVATARS).map((expert) => (
          <ExpertAvatar
            key={expert.id}
            name={expert.name}
            avatarUrl={expert.avatarUrl}
            size={AVATAR_SIZE}
            className="rounded-full ring-2 ring-[#EDE8F1]"
          />
        ))}
      </span>
      <span className="min-w-0">
        {formatReadyLine(experts.map((expert) => expert.name))}
      </span>
    </p>
  );
}
