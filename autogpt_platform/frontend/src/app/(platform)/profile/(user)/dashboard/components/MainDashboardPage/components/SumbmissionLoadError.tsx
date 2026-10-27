import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { InboxIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function SubmissionLoadError() {
  return (
    <div
      // TODO(tokens): no exact role (border-neutral-200 #E5E5E5, dark:border-neutral-700, dark:bg-neutral-800/50)
      className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-neutral-200 bg-background dark:border-neutral-700 dark:bg-neutral-800/50"
    >
      <div className="flex flex-col items-center gap-4 text-center">
        <div
          // TODO(tokens): no exact role (bg-red-100 #FDECEC, dark:bg-red-900/20)
          className="rounded-full bg-red-100 p-3 dark:bg-red-900/20"
        >
          <Icon
            icon={InboxIcon}
            // TODO(tokens): no fitting role for text-red-600 #D93636; #D93636 is error-hover, a hover fill; no exact role (dark:text-red-400)
            className="h-8 w-8 text-red-600 dark:text-red-400"
          />
        </div>
        <div className="space-y-2">
          <Text
            variant="large-medium"
            // TODO(tokens): no exact role (text-neutral-900 #171717, dark:text-neutral-100)
            className="text-neutral-900 dark:text-neutral-100"
          >
            Failed to load workflows
          </Text>
          <Text
            variant="body"
            // TODO(tokens): no exact role (text-neutral-600 #525252, dark:text-neutral-400)
            className="text-neutral-600 dark:text-neutral-400"
          >
            Something went wrong while loading your submitted workflows.
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
