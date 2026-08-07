import Link from "next/link";
import type { BriefingResponse } from "@/app/api/__generated__/models/briefingResponse";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import type { BriefingDecisionItem } from "@/app/api/__generated__/models/briefingDecisionItem";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatBriefingDate } from "./helpers";

interface Props {
  briefing: BriefingResponse;
}

export function BriefingCard({ briefing }: Props) {
  const { run_items, decision_items } = briefing.content;
  const foundItems = run_items.filter((item) => item.summary);

  return (
    <Card className="mb-5 space-y-5 text-left">
      <Text variant="small" className="text-zinc-500">
        {formatBriefingDate(briefing.briefing_date)}
      </Text>

      {run_items.length > 0 ? (
        <section className="flex flex-col gap-3">
          <Text variant="h5">What ran</Text>
          <div className="flex flex-col gap-2">
            {run_items.map((item) => (
              <RunRow key={item.execution_id} item={item} />
            ))}
          </div>
        </section>
      ) : null}

      {foundItems.length > 0 ? (
        <section className="flex flex-col gap-3">
          <Text variant="h5">What was found</Text>
          <div className="flex flex-col gap-2">
            {foundItems.map((item) => (
              <Text
                key={item.execution_id}
                variant="body"
                className="text-zinc-700"
              >
                {item.summary}
              </Text>
            ))}
          </div>
        </section>
      ) : null}

      {decision_items.length > 0 ? (
        <section className="flex flex-col gap-3">
          <Text variant="h5">
            Needs your decision ({decision_items.length})
          </Text>
          <div className="flex flex-col gap-2">
            {decision_items.map((item) => (
              <DecisionRow key={item.node_exec_id} item={item} />
            ))}
          </div>
        </section>
      ) : null}
    </Card>
  );
}

function ExpertAvatar({
  name,
  avatarUrl,
}: {
  name: string | null;
  avatarUrl: string | null;
}) {
  if (!name) return null;
  return (
    <Avatar className="h-8 w-8 shrink-0">
      {avatarUrl ? <AvatarImage src={avatarUrl} alt={name} /> : null}
      <AvatarFallback>{name}</AvatarFallback>
    </Avatar>
  );
}

function RunRow({ item }: { item: BriefingRunItem }) {
  const isFailed = item.status !== "COMPLETED";
  const subtitle = [item.expert_name, isFailed ? "Failed" : "Completed"]
    .filter(Boolean)
    .join(" · ");

  const body = (
    <>
      <ExpertAvatar
        name={item.expert_name}
        avatarUrl={item.expert_avatar_url}
      />
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          className={cn("truncate", isFailed && "text-destructive")}
        >
          {item.agent_name}
        </Text>
        <Text
          variant="small"
          className={cn(
            "truncate text-zinc-500",
            isFailed && "text-destructive",
          )}
        >
          {subtitle}
        </Text>
      </div>
    </>
  );

  if (!item.link) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-zinc-200 p-3">
        {body}
      </div>
    );
  }

  return (
    <Link
      href={item.link}
      className="flex items-center gap-3 rounded-xl border border-zinc-200 p-3"
    >
      {body}
    </Link>
  );
}

function DecisionRow({ item }: { item: BriefingDecisionItem }) {
  return (
    <Link
      href={item.link}
      className="flex items-center gap-3 rounded-xl border border-zinc-200 p-3"
    >
      <ExpertAvatar
        name={item.expert_name}
        avatarUrl={item.expert_avatar_url}
      />
      <Text variant="body-medium" className="min-w-0 flex-1 truncate">
        {item.title}
      </Text>
    </Link>
  );
}
