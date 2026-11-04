import { formatCost, shortAgo } from "../../../../data/helpers";
import { TaskAssigneeAvatar } from "../../../TaskAssigneeAvatar/TaskAssigneeAvatar";
import { MessageResponse } from "@/components/organisms/Message/Message";
import type { Checkpoint } from "./helpers";
import styles from "./trail.module.css";

interface Props {
  checkpoint: Checkpoint;
  first: boolean;
  last: boolean;
  /** Each turn's cost, shown only under Show work. */
  showCost?: boolean;
}

/** One stop on the line: its dot, who did it, what happened, and when. */
export function CheckpointRow({
  checkpoint,
  first,
  last,
  showCost = false,
}: Props) {
  const live = checkpoint.status === "running";

  return (
    <li
      className={styles.row}
      data-status={checkpoint.status}
      data-first={first ? "" : undefined}
      data-last={last ? "" : undefined}
    >
      <span aria-hidden className={styles.rail}>
        <span className={styles.line} />
        <span className={styles.dot} />
      </span>
      <span aria-hidden className={styles.icon}>
        <TaskAssigneeAvatar assignee={checkpoint.actor} size={18} />
      </span>
      <div className={styles.name}>
        <MessageResponse className="size-auto break-words">
          {checkpoint.title}
        </MessageResponse>
      </div>
      {checkpoint.status === "failed" ? (
        <span className={styles.badge}>Failed</span>
      ) : null}
      <span className={styles.meta}>
        {showCost && checkpoint.cost ? (
          <span className={styles.cost}>{formatCost(checkpoint.cost)}</span>
        ) : null}
        <span className={styles.time}>
          {live ? "now" : shortAgo(checkpoint.at)}
        </span>
      </span>
    </li>
  );
}
