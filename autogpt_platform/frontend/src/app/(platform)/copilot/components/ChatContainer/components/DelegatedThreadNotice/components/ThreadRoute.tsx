import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

interface Person {
  name: string;
  avatarUrl: string | null;
  color?: string | null;
}

interface Props {
  delegator: Person | null;
  expert: Person | null;
}

/** Otto (or the teammate who delegated) → the expert this thread belongs to. */
export function ThreadRoute({ delegator, expert }: Props) {
  return (
    <div className="flex shrink-0 items-center gap-2.5" aria-hidden="true">
      {delegator ? (
        <ExpertAvatar
          name={delegator.name}
          avatarUrl={delegator.avatarUrl}
          color={delegator.color}
          size={24}
        />
      ) : (
        <AutopilotAvatar size={24} />
      )}
      <Icon icon={ArrowRight01Icon} size={14} className="text-zinc-900" />
      <ExpertAvatar
        name={expert?.name ?? null}
        avatarUrl={expert?.avatarUrl ?? null}
        color={expert?.color}
        size={24}
      />
    </div>
  );
}
