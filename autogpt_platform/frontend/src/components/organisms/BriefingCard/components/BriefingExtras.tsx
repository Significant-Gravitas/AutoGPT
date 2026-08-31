import {
  Clock01Icon,
  Copy01Icon,
  UserAdd01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { BriefingContent } from "@/app/api/__generated__/models/briefingContent";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { HireRow } from "./HireRow";
import { MergeRow } from "./MergeRow";
import { NudgeRow } from "./NudgeRow";

interface Props {
  content: BriefingContent;
}

// The recap's follow-ups: what the user is holding up, likely duplicates, and
// a hire Autopilot's workload suggests. Each section only earns its space when
// the backend actually sent items for it.
export function BriefingExtras({ content }: Props) {
  const nudges = content.nudge_items ?? [];
  const merges = content.merge_items ?? [];
  const hires = content.hire_items ?? [];

  if (nudges.length === 0 && merges.length === 0 && hires.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 flex flex-col gap-4">
      {nudges.length > 0 ? (
        <Section title="Waiting on you" icon={Clock01Icon}>
          {nudges.map((item) => (
            <NudgeRow key={item.task_id} item={item} />
          ))}
        </Section>
      ) : null}

      {merges.length > 0 ? (
        <Section title="Possible duplicates" icon={Copy01Icon}>
          {merges.map((item, index) => (
            <MergeRow key={item.task_ids.join("-") || index} item={item} />
          ))}
        </Section>
      ) : null}

      {hires.length > 0 ? (
        <Section title="Hire recommendation" icon={UserAdd01Icon}>
          {hires.map((item) => (
            <HireRow key={item.template_id} item={item} />
          ))}
        </Section>
      ) : null}
    </div>
  );
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: IconSvgElement;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 px-2">
        <Icon icon={icon} size={16} className="text-zinc-400" />
        <Text variant="body" className="text-zinc-700">
          {title}
        </Text>
      </div>
      <div className="overflow-hidden rounded-3xl bg-white shadow-zinc-950 smooth-shadow-ring-sm">
        <ul className="divide-y divide-zinc-100">{children}</ul>
      </div>
    </div>
  );
}
