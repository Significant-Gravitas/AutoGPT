"use client";

import { motion } from "framer-motion";
import type { Task } from "../../../../data/types";
import { useInboxMotion } from "../../../InboxList/useInboxMotion";
import {
  TaskAssigneeAvatar,
  assigneeLabel,
} from "../../../TaskAssigneeAvatar/TaskAssigneeAvatar";
import { approveLabel, cardLine } from "../../../CardApprovals/helpers";
import styles from "../../board.module.css";
import { StateBadge } from "../StateBadge";
import { StepTrail } from "./StepTrail";
import { useBoardCard } from "./useBoardCard";
import { useCardActions } from "./useCardActions";

interface Props {
  task: Task;
  isOpen: boolean;
  selected: boolean;
  onToggle(): void;
  onOpen(): void;
}

/** One task on the board. A click unfolds it in place: who works it, its
 *  last steps, what it needs from you, and Cancel / Open. */
export function BoardCard({ task, isOpen, selected, onToggle, onOpen }: Props) {
  const motionProps = useInboxMotion(`board-task-${task.id}`);
  const card = useBoardCard(task);
  const act = useCardActions(task);
  const foldID = `board-card-${task.id}`;

  return (
    <motion.li
      {...motionProps}
      data-testid="inbox-row"
      data-state={task.state}
      data-open={isOpen ? "" : undefined}
      data-selected={selected ? "" : undefined}
      className={styles.card}
    >
      <button
        type="button"
        id={`inbox-row-${task.id}`}
        aria-expanded={isOpen}
        aria-controls={foldID}
        onClick={onToggle}
        className={styles.main}
      >
        <span className={styles.titleRow}>
          <span className={styles.avatar}>
            <TaskAssigneeAvatar assignee={task.assignee} size={24} />
          </span>
          <span className={styles.title} title={task.title}>
            {task.title}
          </span>
        </span>
        {task.now ? <span className={styles.desc}>{task.now}</span> : null}
        <span className={styles.foot}>
          <StateBadge state={task.state} clock={card.clock} />
          {task.metrics.tool_calls ? (
            <span
              className={styles.meta}
              title={`Tool calls: ${task.metrics.tool_calls}`}
            >
              <WrenchGlyph />
              {task.metrics.tool_calls}
            </span>
          ) : null}
        </span>
      </button>

      <div
        id={foldID}
        className={styles.fold}
        data-open={isOpen ? "" : undefined}
        inert={!isOpen}
      >
        <div className={styles.clip}>
          <div className={styles.more}>
            <div className={styles.who}>
              {task.assignee ? assigneeLabel(task.assignee) : "Otto"}
            </div>
            {card.steps.length ? <StepTrail steps={card.steps} /> : null}
            {act.cards.length ? (
              <div className={styles.hire}>
                {act.cards.map((c) => (
                  <div key={c.confirmation_id}>
                    <div className={styles.hireName}>{cardLine(c)}</div>
                    {c.tagline ? (
                      <div className={styles.hireLine}>{c.tagline}</div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : act.question ? (
              <div className={styles.question}>{act.question}</div>
            ) : null}
            <div className={styles.actions}>
              {act.canCancel ? (
                <button
                  type="button"
                  className={styles.btn}
                  disabled={act.busy}
                  onClick={act.cancel}
                >
                  Cancel
                </button>
              ) : null}
              <button type="button" className={styles.btn} onClick={onOpen}>
                Open
              </button>
              {act.cards.length ? (
                <>
                  <button
                    type="button"
                    className={styles.btn}
                    disabled={act.busy}
                    onClick={act.decline}
                  >
                    Decline
                  </button>
                  <button
                    type="button"
                    className={`${styles.btn} ${styles.primary}`}
                    disabled={act.busy}
                    onClick={act.approve}
                  >
                    {approveLabel(act.cards.length)}
                  </button>
                </>
              ) : act.action ? (
                <button
                  type="button"
                  className={`${styles.btn} ${styles.primary}`}
                  disabled={act.busy}
                  onClick={act.runAction}
                >
                  {act.action.label}
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </motion.li>
  );
}

function WrenchGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
      <path
        d="M7.5 1.7a2.5 2.5 0 0 0-2.3 3.4L1.8 8.5a1 1 0 0 0 1.5 1.5l3.4-3.4a2.5 2.5 0 0 0 3.4-2.3L8.7 5.4l-1.4-.4-.4-1.4 1.1-1.5-.5-.4Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  );
}
