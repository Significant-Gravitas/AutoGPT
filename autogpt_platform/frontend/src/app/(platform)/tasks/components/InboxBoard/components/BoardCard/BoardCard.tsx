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
import { RenameField } from "../../../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../../../TaskMenu/TaskMenu";
import { StateBadge } from "../StateBadge";
import { useBoardCard } from "./useBoardCard";
import { useCardActions } from "./useCardActions";

interface Props {
  task: Task;
  isOpen: boolean;
  selected: boolean;
  onToggle(): void;
  onOpen(): void;
}

/** One task on the board. A click unfolds it in place: who works it, the
 *  full line, and the buttons for what it needs from you. */
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
      {card.isRenaming ? (
        <div className={styles.renaming}>
          <RenameField
            taskId={task.id}
            title={task.title}
            onDone={card.stopRename}
          />
        </div>
      ) : (
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
          {card.line ? (
            <span className={styles.desc} title={card.line}>
              {card.line}
            </span>
          ) : null}
          <span className={styles.foot}>
            <StateBadge state={task.state} age={card.age} />
          </span>
        </button>
      )}
      <div className={styles.menu}>
        <TaskMenu
          taskId={task.id}
          title={task.title}
          onRename={card.startRename}
        />
      </div>

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
