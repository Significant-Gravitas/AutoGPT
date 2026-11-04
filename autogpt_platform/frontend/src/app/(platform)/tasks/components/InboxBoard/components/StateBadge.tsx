import { statusOf } from "../../../data/helpers";
import type { TaskState } from "../../../data/types";
import styles from "../board.module.css";

interface Props {
  state: TaskState;
  age: string | null;
}

/** The card's one status mark: a turning ring while it works, a triangle
 *  when it waits on you, a red mark when it failed, then the word and age. */
export function StateBadge({ state, age }: Props) {
  const status = statusOf(state);
  return (
    <span className={styles.state} data-testid="task-status">
      <span aria-hidden className={styles.stateIcon} data-state={state}>
        <StateGlyph state={state} />
      </span>
      <span className={styles.stateWord} data-state={state}>
        {status.word}
      </span>
      {age ? <span className={styles.stateAge}>· {age}</span> : null}
    </span>
  );
}

function StateGlyph({ state }: { state: TaskState }) {
  switch (state) {
    case "working":
      return (
        <svg width="12" height="12" viewBox="0 0 12 12">
          <circle
            cx="6"
            cy="6"
            r="4.5"
            fill="none"
            stroke="currentColor"
            strokeOpacity="0.25"
            strokeWidth="1.6"
          />
          <path
            d="M10.5 6A4.5 4.5 0 0 0 6 1.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
      );
    case "needs_you":
      return (
        <svg width="12" height="12" viewBox="0 0 12 12">
          <path d="M6 1.8 10.8 10H1.2L6 1.8Z" fill="currentColor" />
        </svg>
      );
    case "stuck":
      return (
        <svg width="12" height="12" viewBox="0 0 12 12">
          <circle cx="6" cy="6" r="4.8" fill="currentColor" />
          <path
            d="M6 3.5v3M6 8.4v.1"
            stroke="var(--board-card)"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      );
    case "done":
      return (
        <svg width="12" height="12" viewBox="0 0 12 12">
          <circle cx="6" cy="6" r="3.4" fill="currentColor" />
        </svg>
      );
    case "stopped":
      return (
        <svg width="12" height="12" viewBox="0 0 12 12">
          <circle
            cx="6"
            cy="6"
            r="4.2"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
          />
          <path
            d="M3.3 8.7 8.7 3.3"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      );
  }
}
