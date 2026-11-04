"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import type { Contract } from "@/app/api/__generated__/models/contract";
import type { ReactNode } from "react";
import { formatMetrics } from "../../../../data/helpers";
import type { TaskDetail, TaskState } from "../../../../data/types";
import { ActivityCheckpoints } from "../ActivityCheckpoints/ActivityCheckpoints";
import { DoneWhen } from "../DoneWhen/DoneWhen";
import { Fold } from "../Fold/Fold";
import { GoalLine } from "../GoalLine/GoalLine";
import { Progress } from "../Progress/Progress";
import { useShowWork } from "./useShowWork";

interface Props {
  task: TaskDetail;
  contract: Contract | null;
  state: TaskState;
  now: string;
  running: boolean;
  /** The card at the top already shows the proof. */
  proofInCard: boolean;
}

/** Everything a person wouldn't say out loud, folded away: the goal and its
 *  checks, the plan, every step with its cost, what was used, attachments
 *  and the work chat. */
export function ShowWork({
  task,
  contract,
  state,
  now,
  running,
  proofInCard,
}: Props) {
  const work = useShowWork();
  const tools = task.inbox.tools_used;
  const bodyID = `show-work-${task.id}`;
  const metrics = task.metrics;
  const spent = metrics.tokens > 0 || metrics.cost_usd > 0;

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
          {contract ? (
            <GoalLine
              title={task.title}
              goal={contract.goal}
              previousGoal={contract.previous_goal}
            />
          ) : null}
          {contract ? (
            <DoneWhen
              taskId={task.id}
              contract={contract}
              withProof={!proofInCard}
            />
          ) : null}
          {contract && state !== "working" ? (
            <Progress plan={contract.plan ?? []} now={now} state={state} />
          ) : null}
          <ActivityCheckpoints
            taskId={task.id}
            title={task.title}
            running={running}
            all
          />
          {spent ? (
            <Block label="Cost">
              <Text variant="small" tone="quiet" tabular>
                {formatMetrics(metrics)}
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
