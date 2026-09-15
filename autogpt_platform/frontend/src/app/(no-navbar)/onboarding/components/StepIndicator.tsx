import { cn } from "@/lib/utils";

interface Props {
  totalSteps: number;
  currentStep: number;
}

export function StepIndicator({ totalSteps, currentStep }: Props) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: totalSteps }, (_, i) => (
        <div
          key={i}
          className={cn(
            "h-1.5 rounded-full transition-all",
            i + 1 === currentStep ? "w-4 bg-zinc-900" : "w-1.5 bg-zinc-300",
          )}
        />
      ))}
    </div>
  );
}
