"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { CheckpointRow } from "./CheckpointRow";
import styles from "./trail.module.css";
import { useActivityCheckpoints } from "./useActivityCheckpoints";

interface Props {
  taskId: string;
  title: string;
  running: boolean;
  /** Every step with its cost, for Show work; otherwise checkpoints only. */
  all?: boolean;
  resultSummary?: string | null;
}

/** What happened on the task, oldest first, as stops on one line: who did
 *  what and when. The newest one is blue, and shimmers while the work goes
 *  on. Show work passes `all` for every step and what each one cost. */
export function ActivityCheckpoints({
  taskId,
  title,
  running,
  all = false,
  resultSummary,
}: Props) {
  const activity = useActivityCheckpoints({
    taskId,
    running,
    all,
    resultSummary,
  });
  const count = activity.checkpoints.length;
  const label = all ? "Every step" : "Activity";

  return (
    <section aria-label={label} className={all ? undefined : "px-1.5 py-4"}>
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
          <span className={styles.title}>{label}</span>
          {count && all ? (
            <span className={styles.sum}>
              {count} {count === 1 ? "step" : "steps"}
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
          <ol aria-label={`${label} on ${title}`} className={styles.list}>
            {activity.checkpoints.map((checkpoint, i) => (
              <CheckpointRow
                key={checkpoint.id}
                checkpoint={checkpoint}
                showCost={all}
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
