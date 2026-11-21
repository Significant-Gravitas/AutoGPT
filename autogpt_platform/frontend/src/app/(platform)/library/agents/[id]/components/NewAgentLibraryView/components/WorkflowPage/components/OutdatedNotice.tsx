import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { stashCopilotPrompt } from "@/services/copilot/prompt-handoff";
import { useRouter } from "next/navigation";

interface Props {
  workflowName: string;
  count: number;
  canAsk: boolean;
}

export function OutdatedNotice({ workflowName, count, canAsk }: Props) {
  const router = useRouter();

  function handleAsk() {
    stashCopilotPrompt({
      prompt: `The page for my "${workflowName}" workflow is out of date. Please update it to fit the workflow as it is now.`,
      autosubmit: false,
    });
    router.push("/home");
  }

  return (
    <div
      role="status"
      className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-warning-surface p-4"
    >
      <Text variant="body">
        This page is out of date:{" "}
        {count === 1
          ? "one section no longer fits the workflow"
          : `${count} sections no longer fit the workflow`}{" "}
        and is hidden. Ask Otto to update it.
      </Text>
      {canAsk ? (
        <Button variant="outline" size="small" onClick={handleAsk}>
          Ask Otto to update it
        </Button>
      ) : null}
    </div>
  );
}
