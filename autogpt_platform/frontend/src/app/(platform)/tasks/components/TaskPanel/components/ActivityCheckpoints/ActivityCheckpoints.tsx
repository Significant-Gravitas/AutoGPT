"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { formatCost } from "../../../../data/helpers";
import { CheckpointRow } from "./CheckpointRow";
import styles from "./trail.module.css";
import { useActivityCheckpoints } from "./useActivityCheckpoints";

interface Props {
  taskId: string;
  title: string;
  running: boolean;
}

/** What happened on the task, oldest first, as stops on one line: who did
 *  what, when, and what each work turn cost. The newest one is blue, and
 *  shimmers while the work goes on. */
export function ActivityCheckpoints({ taskId, title, running }: Props) {
  const activity = useActivityCheckpoints(taskId, running);
  const count = activity.checkpoints.length;

  return (
    <section aria-label="Activity" className="px-1.5 py-4">
      <div className={styles.tray}>
        <div className={styles.head}>
          <span aria-hidden className={styles.mark}>
            <svg width="14" height="14" viewBox="0 0 14 14">
              <circle cx="7" cy="3" r="1.8" fill="currentColor" />
              <circle cx="7" cy="11" r="1.8" fill="currentColor" />
              <path
                d="M7 4.8v4.4"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </span>
          <span className={styles.title}>Activity</span>
          <span className={styles.task}>{title}</span>
          {count ? (
            <span className={styles.sum}>
              {count} {count === 1 ? "step" : "steps"}
              {activity.spent > 0 ? ` · ${formatCost(activity.spent)}` : ""}
            </span>
          ) : null}
        </div>
        {activity.hasMore ? (
          <button
            type="button"
            className={styles.more}
            disabled={activity.isLoadingMore}
            onClick={activity.loadMore}
          >
            {activity.isLoadingMore ? "Loading…" : "Show earlier activity"}
          </button>
        ) : null}
        {activity.isLoading ? (
          <div className="flex flex-col gap-2 rounded-card bg-surface p-3">
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-2/3" />
          </div>
        ) : count ? (
          <ol aria-label={`Activity on ${title}`} className={styles.list}>
            {activity.checkpoints.map((checkpoint, i) => (
              <CheckpointRow
                key={checkpoint.id}
                checkpoint={checkpoint}
                first={i === 0}
                last={i === count - 1}
              />
            ))}
          </ol>
        ) : (
          <p className={styles.empty}>No activity yet.</p>
        )}
      </div>
    </section>
  );
}
