"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import type { ExpertOption, TaskDetail } from "../../../../data/types";
import { ActivityFeed } from "../ActivityFeed/ActivityFeed";
import { Fold } from "../Fold/Fold";
import { useShowWork } from "./useShowWork";

interface Props {
  task: TaskDetail;
  experts: ExpertOption[];
}

/** Everything behind the summary, folded away: the brief, what was used,
 *  attachments, the work chat and the full log. */
export function ShowWork({ task, experts }: Props) {
  const work = useShowWork(task.id);
  const tools = task.inbox.tools_used;
  const bodyID = `show-work-${task.id}`;

  return (
    <section aria-label="Show work" className="px-1.5 py-3">
      <button
        type="button"
        aria-expanded={work.open}
        aria-controls={bodyID}
        onClick={work.toggle}
        className="flex w-full items-center gap-1.5 rounded-control py-1 text-left focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
      >
        <Text as="span" variant="inbox-medium" tone="quiet">
          Show work
        </Text>
        <Icon
          icon={ArrowDown01Icon}
          size={14}
          tone="faint"
          motion="transform"
          className={cn(work.open && "rotate-180")}
          aria-hidden
        />
      </button>
      <Fold open={work.open} id={bodyID}>
        <div className="flex flex-col gap-4 pt-3">
          {task.description ? (
            <Block label="Brief">
              <Text
                variant="inbox"
                tone="text"
                className="break-words whitespace-pre-wrap"
              >
                {task.description}
              </Text>
            </Block>
          ) : null}
          {tools.length ? (
            <Block label="Used">
              <Text variant="small" tone="quiet">
                {tools
                  .map((t) =>
                    t.count > 1 ? `${t.title} ×${t.count}` : t.title,
                  )
                  .join(" · ")}
              </Text>
            </Block>
          ) : null}
          {task.artifacts.length ? (
            <Block label="Attachments">
              <ul className="flex flex-col gap-1">
                {task.artifacts.map((artifact) => (
                  <li key={artifact.id}>
                    {artifact.url ? (
                      <Link
                        href={artifact.url}
                        isExternal={!artifact.url.startsWith("/")}
                      >
                        {artifact.title}
                      </Link>
                    ) : (
                      <Text variant="small" tone="text">
                        {artifact.title}
                      </Text>
                    )}
                  </li>
                ))}
              </ul>
            </Block>
          ) : null}
          {task.thread_session_id ? (
            <Link
              href={`/copilot?sessionId=${task.thread_session_id}`}
              className="self-start"
            >
              Open the work chat
            </Link>
          ) : null}
          <Block label="Activity">
            <ActivityFeed
              items={work.items}
              isLoading={work.isLoading}
              experts={experts}
              hasMore={work.hasMore}
              isLoadingMore={work.isLoadingMore}
              onLoadMore={work.loadMore}
            />
          </Block>
        </div>
      </Fold>
    </section>
  );
}

function Block({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <Text as="h4" variant="small" tone="faint">
        {label}
      </Text>
      {children}
    </div>
  );
}
