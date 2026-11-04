import styles from "../../board.module.css";

interface Props {
  steps: { label: string; time: string | null }[];
}

/** The last few steps: small dots joined by short lines, the newest dark. */
export function StepTrail({ steps }: Props) {
  return (
    <ol className={styles.steps}>
      {steps.map((step, i) => (
        <li
          key={`${step.label}-${i}`}
          className={styles.step}
          data-last={i === steps.length - 1 ? "" : undefined}
        >
          <span className={styles.stepLabel}>{step.label}</span>
          {step.time ? (
            <time className={styles.stepTime}>{step.time}</time>
          ) : null}
        </li>
      ))}
    </ol>
  );
}
