interface Props {
  currentStep: number;
  totalSteps: number;
}

export function ProgressBar({ currentStep, totalSteps }: Props) {
  const percent = (currentStep / totalSteps) * 100;

  return (
    <div className="absolute left-0 top-0 h-[0.625rem] w-full bg-neutral-300">
      <div
        className="h-full bg-purple-400 shadow-[0_0_4px_2px_rgba(168,85,247,0.5)] transition-all duration-500 ease-out"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
