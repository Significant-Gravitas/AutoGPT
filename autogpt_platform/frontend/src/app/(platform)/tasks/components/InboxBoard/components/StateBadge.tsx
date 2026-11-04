import type { TaskState } from "../../../data/types";
import styles from "../board.module.css";
import { STATE_WORD } from "../helpers";

interface Props {
  state: TaskState;
  clock: string | null;
}

/** The card's state: a turning ring while it runs, a triangle when it
 *  waits on you, a dot when done, then the word and how long. */
export function StateBadge({ state, clock }: Props) {
  return (
    <span className={styles.state}>
      <span aria-hidden className={styles.stateIcon} data-state={state}>
        <StateGlyph state={state} />
      </span>
      <span className={styles.stateWord}>
        {!clock
          ? STATE_WORD[state]
          : state === "done"
            ? `Done in ${clock}`
            : `${STATE_WORD[state]} · ${clock}`}
      </span>
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
            stroke="white"
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
