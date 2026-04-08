interface Props {
  currentStep: number;
  totalSteps: number;
}

export function ProgressBar({ currentStep, totalSteps }: Props) {
  const percent = (currentStep / totalSteps) * 100;

  return (
    <div className="absolute left-0 top-0 h-[3px] w-full bg-neutral-200">
      <div
        className="h-full bg-purple-400 transition-all duration-500 ease-out"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
