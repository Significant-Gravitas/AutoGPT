import { Tray } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";

export function SubmissionLoadError() {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-neutral-200 bg-neutral-50 dark:border-neutral-700 dark:bg-neutral-800/50">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-full bg-red-100 p-3 dark:bg-red-900/20">
          <Tray className="h-8 w-8 text-red-600 dark:text-red-400" />
        </div>
        <div className="space-y-2">
          <Text
            variant="large-medium"
            className="text-neutral-900 dark:text-neutral-100"
          >
            Failed to load agents
          </Text>
          <Text
            variant="body"
            className="text-neutral-600 dark:text-neutral-400"
          >
            Something went wrong while loading your submitted agents.
          </Text>
        </div>
        <Button
          variant="secondary"
          size="small"
          onClick={() => window.location.reload()}
        >
          Try again
        </Button>
      </div>
    </div>
  );
}
