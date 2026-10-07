import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  expert: Expert;
  isRetrying: boolean;
  onRetry: () => void;
}

export function SetupStatus({ expert, isRetrying, onRetry }: Props) {
  if (expert.setup_status === "installing") {
    return (
      <div
        role="status"
        className="mx-4 mb-3 rounded-lg bg-sky-50 px-3 py-2 ring-1 ring-sky-200 ring-inset"
      >
        <Text variant="body" className="text-sky-800">
          Setting up {expert.name}&apos;s skills and workflows…
        </Text>
      </div>
    );
  }
  if (expert.setup_status !== "failed") return null;
  const failures = expert.setup_failures ?? [];
  return (
    <div
      role="alert"
      className="mx-4 mb-3 flex items-center justify-between gap-2 rounded-lg bg-yellow-50 px-3 py-2 ring-1 ring-yellow-200 ring-inset"
    >
      <Text variant="body" className="text-yellow-700">
        {failures.length
          ? `Couldn't install: ${failures.join(", ")}`
          : "Setup didn't finish"}
      </Text>
      {expert.source_template_id ? (
        <Button
          variant="secondary"
          size="small"
          loading={isRetrying}
          onClick={onRetry}
        >
          Retry setup
        </Button>
      ) : null}
    </div>
  );
}
