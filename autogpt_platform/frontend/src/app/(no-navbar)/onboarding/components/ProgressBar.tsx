interface Props {
  currentStep: number;
  totalSteps: number;
}

export function ProgressBar({ currentStep, totalSteps }: Props) {
  // The progress bar tracks the user-interactive steps PLUS the trailing
  // Preparing step (which doesn't render the bar itself). Using
  // `totalSteps + 1` keeps the last interactive step (e.g. Subscription)
  // at 80% rather than maxing out at 100% before the user is actually done.
  const percent = (currentStep / (totalSteps + 1)) * 100;

  return (
    <div className="absolute left-0 top-0 h-0.5 w-full bg-zinc-100">
      <div
        className="h-full bg-zinc-900 transition-all duration-500 ease-out"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
