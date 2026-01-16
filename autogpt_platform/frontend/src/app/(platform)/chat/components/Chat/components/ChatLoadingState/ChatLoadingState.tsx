import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";

export interface ChatLoadingStateProps {
  message?: string;
  className?: string;
}

export function ChatLoadingState({ className }: ChatLoadingStateProps) {
  return (
    <div
      className={cn("flex flex-1 items-center justify-center p-6", className)}
    >
      <div className="flex flex-col items-center gap-4 text-center">
        <LoadingSpinner />
      </div>
    </div>
  );
}
