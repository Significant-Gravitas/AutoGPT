import {
  BubbleChatIcon,
  CheckmarkCircle02Icon,
  File01Icon,
  Link01Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import NextLink from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { TaskArtifact, TaskDetail } from "../../../../data/types";

interface Props {
  task: TaskDetail;
}

const ARTIFACT_ICONS: Record<TaskArtifact["kind"], IconSvgElement> = {
  file: File01Icon,
  chat: BubbleChatIcon,
  run: PlayIcon,
  link: Link01Icon,
};

export function TaskOutcome({ task }: Props) {
  if (!task.outcome_summary && !task.artifacts.length) return null;

  return (
    <section aria-label="Outcome" className="animate-inbox-enter px-5 py-4">
      <div className="flex flex-col gap-2.5 rounded-card bg-status-ok-surface p-4">
        <div className="flex items-center gap-1.5">
          <Icon icon={CheckmarkCircle02Icon} tone="ok" size={14} aria-hidden />
          <Text as="h3" variant="inbox-medium" tone="ok">
            Outcome
          </Text>
        </div>
        {task.outcome_summary ? (
          <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
            {task.outcome_summary}
          </Text>
        ) : null}
        {task.artifacts.length ? (
          <ul className="flex flex-col divide-y divide-border-hairline overflow-hidden rounded-card bg-surface shadow-card">
            {task.artifacts.map((artifact) => (
              <li key={artifact.id}>
                <ArtifactLink artifact={artifact} />
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  );
}

function ArtifactLink({ artifact }: { artifact: TaskArtifact }) {
  const content = (
    <>
      <Icon
        icon={ARTIFACT_ICONS[artifact.kind]}
        tone="quiet"
        size={14}
        aria-hidden
      />
      <Text variant="inbox" tone="text" as="span" className="truncate">
        {artifact.title}
      </Text>
    </>
  );
  const className =
    "flex min-h-9 items-center gap-2.5 px-3 transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset";
  if (!artifact.url || !/^(https?:\/\/|\/)/.test(artifact.url)) {
    return (
      <div className="flex min-h-9 items-center gap-2.5 px-3">{content}</div>
    );
  }
  const isExternal = /^https?:\/\//.test(artifact.url);
  return (
    <NextLink
      href={artifact.url}
      className={className}
      target={isExternal ? "_blank" : undefined}
      rel={isExternal ? "noopener noreferrer" : undefined}
    >
      {content}
    </NextLink>
  );
}
