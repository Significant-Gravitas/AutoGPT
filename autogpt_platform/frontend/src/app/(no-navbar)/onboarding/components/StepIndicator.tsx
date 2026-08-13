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
            "h-2 rounded-full transition-all",
            i + 1 === currentStep ? "w-6 bg-foreground" : "w-2 bg-gray-300",
          )}
        />
      ))}
    </div>
  );
}
