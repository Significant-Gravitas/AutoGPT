import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  message: string;
  totalSteps: number;
  showMessage?: boolean;
}

export function WizardPlaceholder({
  message,
  totalSteps,
  showMessage = false,
}: Props) {
  return (
    <div
      className="flex min-h-screen w-full flex-col items-center"
      data-testid="onboarding-placeholder"
    >
      <div className="absolute top-0 left-0 h-0.5 w-full bg-surface-hover" />
      <div className="flex w-full min-w-0 flex-1 items-center justify-center pt-16 pb-8">
        <div className="flex w-full max-w-md flex-col items-center gap-4 px-4">
          <Skeleton variant="translucent" className="h-8 w-3/4" aria-hidden />
          <Skeleton variant="translucent" className="h-4 w-full" aria-hidden />
          <Skeleton variant="translucent" className="h-4 w-5/6" aria-hidden />
          <Text
            variant="body"
            role="status"
            className={cn(!showMessage && "sr-only")}
          >
            {message}
          </Text>
          <Skeleton
            variant="translucent"
            className="mt-4 h-10 w-40 rounded-full"
            aria-hidden
          />
        </div>
      </div>
      <div className="flex items-center gap-2 pb-8" aria-hidden>
        {Array.from({ length: totalSteps }, (_, i) => (
          <Skeleton
            key={i}
            variant="translucent"
            className="h-1.5 w-1.5 rounded-full"
          />
        ))}
      </div>
    </div>
  );
}
