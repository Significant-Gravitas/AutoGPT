import type { HomeBriefing } from "@/app/api/__generated__/models/homeBriefing";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";

interface Props {
  briefing: HomeBriefing;
}

/** The brief and its byline. Otto writes it whatever the team did, so
 *  the author is never one of the experts the paragraph reports on. */
export function BriefingByline({ briefing }: Props) {
  const { author, narrative } = briefing;

  return (
    <div data-testid="briefing-byline" className="px-4 py-3">
      <div className="mb-1.5 flex items-center gap-2">
        <AutopilotAvatar size={20} />
        <Text variant="small-medium" as="span" tone="primary">
          {author.name}
        </Text>
        <span aria-hidden="true" className="text-zinc-300">
          ·
        </span>
        <Text variant="small" as="span" tone="muted">
          {author.role}
        </Text>
      </div>
      <Text variant="body" tone="secondary" className="text-pretty leading-5">
        {narrative}
      </Text>
    </div>
  );
}
