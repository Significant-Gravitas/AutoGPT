"use client";

import { postExperimentalSetSessionComputerUseConsent } from "@/app/api/__generated__/endpoints/copilot/copilot";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ViewIcon } from "@hugeicons/core-free-icons";
import { useMutation } from "@tanstack/react-query";
import {
  type LocalPCExecutorStatus,
  useLocalPCExecutor,
} from "../../hooks/useLocalPCExecutor";
import { describeComputerUseTools } from "./helpers";

function consentScope(executor: LocalPCExecutorStatus | undefined) {
  return {
    expectedMachineID: executor?.machine_id ?? "",
    expectedDriver: executor?.computer_use_driver ?? "",
    expectedTools: Array.from(
      new Set(executor?.computer_use_tools ?? []),
    ).sort(),
  };
}

interface Props {
  sessionID: string | null;
}

export function LocalPCComputerUseConsent({ sessionID }: Props) {
  const { data: executor, refetch } = useLocalPCExecutor(sessionID);
  const consentMutation = useMutation({
    mutationFn: ({
      approved,
      submittedForSessionID,
      expectedMachineID,
      expectedDriver,
      expectedTools,
    }: {
      approved: boolean;
      submittedForSessionID: string;
      expectedMachineID: string;
      expectedDriver: string;
      expectedTools: string[];
    }) => {
      const request = approved
        ? {
            approved,
            expected_machine_id: expectedMachineID,
            expected_driver: expectedDriver,
            expected_tools: expectedTools,
          }
        : { approved };
      return postExperimentalSetSessionComputerUseConsent(
        submittedForSessionID,
        request,
      );
    },
    onSuccess: async (_, { submittedForSessionID }) => {
      if (submittedForSessionID === sessionID) await refetch();
    },
    onError: async (_, { submittedForSessionID }) => {
      if (submittedForSessionID === sessionID) await refetch();
    },
  });

  const accessLines = describeComputerUseTools(
    executor?.computer_use_tools ?? [],
  );
  const { expectedMachineID, expectedDriver, expectedTools } =
    consentScope(executor);
  const shimHasComputerUse =
    executor?.kind === "shim" && accessLines.length > 0;
  const consentResolved =
    executor?.computer_use_consent === "approved" ||
    executor?.computer_use_consent === "denied";

  if (!sessionID || !shimHasComputerUse || consentResolved) {
    return null;
  }

  const pendingDecision = consentMutation.variables?.approved;

  return (
    <Dialog
      title="Otto is requesting computer access"
      styling={{ maxWidth: "32rem", minWidth: "auto" }}
      controlled={{ isOpen: true, set: async () => {} }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 py-2">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-warning-surface text-warning-foreground">
              <Icon icon={ViewIcon} className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="flex flex-col gap-2">
              <Text variant="body-medium" tone="primary">
                For this chat session, Otto can:
              </Text>
              <ul className="ml-4 list-disc text-body text-foreground-body">
                {accessLines.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
              <Text variant="body-tight" tone="body">
                Otto works mostly in the background through{" "}
                <span className="font-mono text-caption">
                  {executor?.computer_use_driver}
                </span>
                , reading your windows and using their controls directly.
              </Text>
              <Text variant="body-tight" tone="body">
                Before Otto runs a script, saves a screenshot to a file, or
                hands an app or a web page a file from your computer, it asks
                you in this chat, unless the chat runs Unsupervised.
              </Text>
              <Text variant="body-tight" tone="body">
                The shim writes operation records to its local audit log. Review
                them with{" "}
                <span className="font-mono text-caption">
                  autogpt-shim audit tail
                </span>{" "}
                on your machine. You can stop access with Ctrl+C in the
                daemon&apos;s terminal, or disconnect it with{" "}
                <span className="font-mono text-caption">
                  autogpt-shim revoke
                </span>
                .
              </Text>
              <Text variant="body-tight" tone="body">
                This consent applies only to the current chat session. New
                sessions will ask again.
              </Text>
              {consentMutation.isError ? (
                <Text variant="body-tight" role="alert" tone="danger">
                  Could not update computer access. Try again.
                </Text>
              ) : null}
            </div>
          </div>
        </div>
        <Dialog.Footer className="flex-wrap justify-end">
          <Button
            variant="secondary"
            disabled={consentMutation.isPending}
            loading={consentMutation.isPending && pendingDecision === false}
            onClick={() =>
              consentMutation.mutate({
                approved: false,
                submittedForSessionID: sessionID,
                expectedMachineID,
                expectedDriver,
                expectedTools,
              })
            }
          >
            Not this time
          </Button>
          <Button
            variant="primary"
            disabled={consentMutation.isPending}
            loading={consentMutation.isPending && pendingDecision === true}
            onClick={() =>
              consentMutation.mutate({
                approved: true,
                submittedForSessionID: sessionID,
                expectedMachineID,
                expectedDriver,
                expectedTools,
              })
            }
          >
            Allow for this session
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
