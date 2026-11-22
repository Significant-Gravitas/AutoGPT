import { ChartBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/ChartBlockView";
import type { PageExpertActivity } from "@/app/api/__generated__/models/pageExpertActivity";
import type { PageExpertLearned } from "@/app/api/__generated__/models/pageExpertLearned";
import type { PageExpertStatus } from "@/app/api/__generated__/models/pageExpertStatus";
import type { PageExpertWork } from "@/app/api/__generated__/models/pageExpertWork";
import { Text } from "@/components/atoms/Text/Text";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { formatDate } from "@/lib/utils/time";
import NextLink from "next/link";
import { dollars } from "../helpers";

type Section =
  | PageExpertStatus
  | PageExpertWork
  | PageExpertLearned
  | PageExpertActivity;

const STATUS_LABELS: Record<string, string> = {
  ready: "Ready",
  working: "Working",
  needs_setup: "Needs setup",
  paused: "Paused",
  failed: "Last run failed",
};

export function ExpertSection({ section }: { section: Section }) {
  switch (section.kind) {
    case "expert_status":
      return <Status section={section} />;
    case "expert_work":
      return <Work section={section} />;
    case "expert_learned":
      return <Learned section={section} />;
    case "expert_activity":
      return section.block ? (
        <ChartBlockView block={section.block} />
      ) : (
        <Empty>No activity in the last 30 days.</Empty>
      );
  }
}

function Status({ section }: { section: PageExpertStatus }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-3">
      <div className="flex min-w-0 flex-col gap-1">
        <Text variant="body-medium">
          {STATUS_LABELS[section.status] ?? section.status}
        </Text>
        <Text variant="body" tone="secondary">
          {section.detail}
        </Text>
      </div>
      <div className="flex flex-col items-end gap-1">
        <Text variant="body" tabular>
          {dollars(section.spend_cents ?? 0)} this week
        </Text>
        <NextLink href={`/team/${section.expert_id}`} className="underline">
          <Text as="span" variant="small">
            Open {section.expert_name}
          </Text>
        </NextLink>
      </div>
    </div>
  );
}

function Work({ section }: { section: PageExpertWork }) {
  if (!section.items.length) return <Empty>No work yet.</Empty>;
  return (
    <ul className="flex flex-col divide-y divide-separator">
      {section.items.map((item) => (
        <li key={item.execution_id} className="flex flex-col gap-1 py-2">
          <span className="flex items-center justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2">
              <RunStatusBadge status={item.status} />
              <Text variant="body-medium" className="truncate">
                {item.workflow_name}
              </Text>
            </span>
            <Text variant="small" tone="secondary">
              {item.started_at ? formatDate(new Date(item.started_at)) : ""}
            </Text>
          </span>
          {item.summary ? (
            <Text variant="small" tone="secondary" className="line-clamp-2">
              {item.summary}
            </Text>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function Learned({ section }: { section: PageExpertLearned }) {
  if (!section.facts.length) return <Empty>Nothing remembered yet.</Empty>;
  return (
    <ul className="flex list-disc flex-col gap-1.5 pl-5">
      {section.facts.map((fact, index) => (
        <li key={index}>
          <Text variant="body">{fact.fact}</Text>
        </li>
      ))}
    </ul>
  );
}

function Empty({ children }: { children: string }) {
  return (
    <Text variant="body" tone="secondary">
      {children}
    </Text>
  );
}
