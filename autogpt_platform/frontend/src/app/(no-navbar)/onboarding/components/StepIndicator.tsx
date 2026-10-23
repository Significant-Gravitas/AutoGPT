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
            // TODO(tokens): no exact role for bg-zinc-300 (#C5C5C9)
            i + 1 === currentStep ? "w-4 bg-action-hover" : "w-1.5 bg-zinc-300",
          )}
        />
      ))}
    </div>
  );
}
