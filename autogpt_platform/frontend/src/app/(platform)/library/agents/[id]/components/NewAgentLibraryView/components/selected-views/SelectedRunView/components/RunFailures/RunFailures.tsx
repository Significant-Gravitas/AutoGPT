"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import { createIconComponent } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { describeFailedSteps } from "@/lib/run-health";
import { Alert02Icon, CancelCircleIcon } from "@hugeicons/core-free-icons";
import { useRunFailures } from "./useRunFailures";

const WarningIcon = createIconComponent(Alert02Icon);
const FailedIcon = createIconComponent(CancelCircleIcon);

const ERROR_PREVIEW_LIMIT = 300;
const ERROR_CLASS_NAME = "whitespace-pre-wrap break-words";

interface Props {
  run: GraphExecution | GraphExecutionWithNodes;
}

export function RunFailures({ run }: Props) {
  const { steps, failedStepCount, runError, isLoadingNames } =
    useRunFailures(run);

  if (failedStepCount === 0 && !runError) return null;

  const runFailed = run.status === AgentExecutionStatus.FAILED;

  return (
    <div className="px-4" data-testid="run-failures">
      <Alert
        variant={runFailed ? "error" : "warning"}
        icon={runFailed ? FailedIcon : WarningIcon}
        className="[&>svg]:top-3 [&>svg]:translate-y-0"
      >
        <AlertTitle>
          {failedStepCount > 0
            ? describeFailedSteps(failedStepCount)
            : "This run failed"}
        </AlertTitle>
        <AlertDescription>
          <div className="flex flex-col gap-3">
            {run.status === AgentExecutionStatus.COMPLETED ? (
              <Text variant="small" tone="secondary">
                The run still finished, so its output may be incomplete.
              </Text>
            ) : null}
            {runError ? (
              <ShowMoreText
                variant="small"
                previewLimit={ERROR_PREVIEW_LIMIT}
                tone="emphasis"
                className={ERROR_CLASS_NAME}
              >
                {runError}
              </ShowMoreText>
            ) : steps.length > 0 ? (
              <ul className="flex flex-col gap-3">
                {steps.map((step) => (
                  <li key={step.id} className="flex flex-col gap-1">
                    {step.name ? (
                      <Text variant="body" weight="medium">
                        {step.name}
                      </Text>
                    ) : isLoadingNames ? (
                      <Skeleton className="h-5 w-32" />
                    ) : (
                      <Text variant="body" weight="medium">
                        Unknown step
                      </Text>
                    )}
                    <ShowMoreText
                      variant="small"
                      previewLimit={ERROR_PREVIEW_LIMIT}
                      tone="emphasis"
                      className={ERROR_CLASS_NAME}
                    >
                      {step.error}
                    </ShowMoreText>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </AlertDescription>
      </Alert>
    </div>
  );
}
