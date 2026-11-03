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
    <section
      aria-label="Outcome"
      className="flex flex-col gap-3 rounded-xl border border-border-base bg-success-surface p-4"
    >
      <div className="flex items-center gap-2">
        <Icon icon={CheckmarkCircle02Icon} tone="success" aria-hidden />
        <Text variant="small-semibold" tone="success-strong">
          Outcome
        </Text>
      </div>
      {task.outcome_summary ? (
        <Text
          variant="body-tight"
          tone="primary"
          className="whitespace-pre-wrap"
        >
          {task.outcome_summary}
        </Text>
      ) : null}
      {task.artifacts.length ? (
        <ul className="flex flex-col gap-1">
          {task.artifacts.map((artifact) => (
            <li key={artifact.id}>
              <ArtifactLink artifact={artifact} />
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function ArtifactLink({ artifact }: { artifact: TaskArtifact }) {
  const content = (
    <>
      <Icon icon={ARTIFACT_ICONS[artifact.kind]} tone="muted" aria-hidden />
      <Text variant="body-tight" tone="body" as="span" className="truncate">
        {artifact.title}
      </Text>
    </>
  );
  const className =
    "flex items-center gap-2 rounded-md px-2 py-1 transition-colors hover:bg-surface-hover focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none";
  if (!artifact.url || !/^(https?:\/\/|\/)/.test(artifact.url)) {
    return <div className="flex items-center gap-2 px-2 py-1">{content}</div>;
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
