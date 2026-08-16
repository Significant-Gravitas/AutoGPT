import Link from "next/link";
import type { BriefingResponse } from "@/app/api/__generated__/models/briefingResponse";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { formatBriefingDate, isInternalLink } from "./helpers";

interface Props {
  briefing: BriefingResponse;
}

// The briefing's decision items are deliberately not rendered here: the same
// pending reviews already appear — and are actionable — in the
// needs-attention list directly below this card. They stay in the thread
// markdown, where there is no such list.
export function BriefingCard({ briefing }: Props) {
  const { run_items } = briefing.content;
  const foundItems = run_items.filter((item) => item.summary);

  // A briefing can be all decisions and no terminal runs (a run paused on an
  // approval never completes), which would render as a card containing just
  // a date. The needs-attention list below carries that case on its own.
  if (run_items.length === 0) return null;

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
                {/* Attribution matters once more than one agent reports:
                    mirrors the thread markdown's "**{agent}**: {summary}". */}
                <span className="font-medium">{item.agent_name}</span>:{" "}
                {item.summary}
              </Text>
            ))}
          </div>
        </section>
      ) : null}
    </Card>
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
        size={32}
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

  // Relative paths only: the backend composes these, but nothing else stops
  // a future regression from delivering an absolute or `javascript:` URL to
  // a Next.js <Link>.
  if (!item.link || !isInternalLink(item.link)) {
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
