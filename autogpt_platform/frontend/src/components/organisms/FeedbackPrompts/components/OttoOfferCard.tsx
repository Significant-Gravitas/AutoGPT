import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ArrowRight02Icon, SparklesIcon } from "@hugeicons/core-free-icons";

interface Props {
  /** The open rows of the user's suggested tasks, which an LLM writes from
   *  their onboarding answers (GET /api/onboarding/brain-dump/suggested-tasks).
   *  Empty when onboarding stored none; the card then asks with no list
   *  rather than showing generic examples. */
  starters: string[];
}

export function OttoOfferCard({ starters }: Props) {
  const hasStarters = starters.length > 0;

  return (
    <div className="flex flex-col gap-4 rounded-2xl bg-gradient-to-br from-accent-surface to-surface-muted p-5">
      <div className="flex items-start gap-3">
        <AutopilotAvatar size={40} className="ring-1 ring-border-base" />
        <div className="min-w-0 rounded-2xl rounded-tl-sm bg-surface px-4 py-2.5 shadow-sm">
          <Text variant="body" className="!text-foreground-emphasis">
            {hasStarters
              ? "From what you told me, I'd start with one of these."
              : "Tell me one weekly task and I'll do it with you now."}
          </Text>
        </div>
      </div>
      {hasStarters ? (
        <>
          <div className="flex flex-col gap-1.5 pl-[3.25rem]">
            {starters.map((starter) => (
              <button
                key={starter}
                type="button"
                className="group flex items-center justify-between gap-3 rounded-xl bg-surface/70 px-4 py-2.5 text-left text-body text-foreground-emphasis transition-colors hover:bg-surface hover:shadow-sm"
              >
                {starter}
                <Icon
                  icon={ArrowRight02Icon}
                  size={16}
                  className="shrink-0 text-foreground-tertiary transition-transform group-hover:translate-x-0.5"
                />
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 pl-[3.25rem] text-foreground-tertiary">
            <Icon icon={SparklesIcon} size={14} />
            <Text variant="small" className="!text-foreground-tertiary">
              Picked for you from your onboarding answers.
            </Text>
          </div>
        </>
      ) : null}
    </div>
  );
}
