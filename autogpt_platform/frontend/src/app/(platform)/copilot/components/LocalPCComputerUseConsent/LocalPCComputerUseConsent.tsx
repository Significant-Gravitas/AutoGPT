"use client";

import { postExperimentalSetSessionComputerUseConsent } from "@/app/api/__generated__/endpoints/copilot/copilot";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { EyeIcon } from "@phosphor-icons/react";
import { useMutation } from "@tanstack/react-query";
import {
  type LocalPCExecutorStatus,
  useLocalPCExecutor,
} from "../../hooks/useLocalPCExecutor";

const KNOWN_COMPUTER_USE_FEATURES = new Set([
  "screenshot",
  "input",
  "windows",
  "apps",
  "clipboard",
  "permissions",
]);

function advertisedComputerUseFeatures(
  executor: LocalPCExecutorStatus | undefined,
) {
  const coarseFeatures = (executor?.computer_use_features_coarse ?? []).filter(
    (feature) => KNOWN_COMPUTER_USE_FEATURES.has(feature),
  );
  const features = new Set(coarseFeatures);
  for (const feature of executor?.computer_use_features ?? []) {
    if (feature === "screenshot" || feature.startsWith("screenshot.")) {
      features.add("screenshot");
    }
    if (
      feature === "input" ||
      feature.startsWith("input.") ||
      feature === "cursor.position"
    ) {
      features.add("input");
    }
    if (feature === "windows" || feature.startsWith("window.")) {
      features.add("windows");
    }
    if (feature === "apps" || feature.startsWith("app.")) {
      features.add("apps");
    }
    if (feature === "clipboard" || feature.startsWith("clipboard.")) {
      features.add("clipboard");
    }
    if (feature === "permissions" || feature.startsWith("permissions.")) {
      features.add("permissions");
    }
  }
  return features;
}

function consentScope(executor: LocalPCExecutorStatus | undefined) {
  return {
    expectedMachineID: executor?.machine_id ?? "",
    expectedFeaturesCoarse: Array.from(
      new Set(executor?.computer_use_features_coarse ?? []),
    ).sort(),
    expectedFeatures: Array.from(
      new Set(executor?.computer_use_features ?? []),
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
      expectedFeaturesCoarse,
      expectedFeatures,
    }: {
      approved: boolean;
      submittedForSessionID: string;
      expectedMachineID: string;
      expectedFeaturesCoarse: string[];
      expectedFeatures: string[];
    }) => {
      const request = approved
        ? {
            approved,
            expected_machine_id: expectedMachineID,
            expected_features_coarse: expectedFeaturesCoarse,
            expected_features: expectedFeatures,
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

  const computerUseFeatures = advertisedComputerUseFeatures(executor);
  const { expectedMachineID, expectedFeaturesCoarse, expectedFeatures } =
    consentScope(executor);
  const shimHasComputerUse =
    executor?.kind === "shim" && computerUseFeatures.size > 0;
  const consentResolved =
    executor?.computer_use_consent === "approved" ||
    executor?.computer_use_consent === "denied";

  if (!sessionID || !shimHasComputerUse || consentResolved) {
    return null;
  }

  const pendingDecision = consentMutation.variables?.approved;

  return (
    <Dialog
      title="Claude is requesting computer access"
      styling={{ maxWidth: "32rem", minWidth: "auto" }}
      controlled={{ isOpen: true, set: async () => {} }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 py-2">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100">
              <EyeIcon
                className="h-5 w-5 text-amber-700"
                weight="fill"
                aria-hidden="true"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Text variant="body" className="font-medium text-neutral-900">
                For this chat session, Claude can:
              </Text>
              <ul className="ml-4 list-disc text-sm text-neutral-700">
                {computerUseFeatures.has("screenshot") ? (
                  <li>Capture screenshots of your screen</li>
                ) : null}
                {computerUseFeatures.has("input") ? (
                  <li>Move the pointer, click, type, scroll, and send keys</li>
                ) : null}
                {computerUseFeatures.has("windows") ? (
                  <li>List and focus open windows</li>
                ) : null}
                {computerUseFeatures.has("apps") ? (
                  <li>List and launch apps</li>
                ) : null}
                {computerUseFeatures.has("clipboard") ? (
                  <li>Read and write your clipboard</li>
                ) : null}
                {computerUseFeatures.has("permissions") ? (
                  <li>Check operating-system permission status</li>
                ) : null}
              </ul>
              <Text variant="body" className="text-sm text-neutral-700">
                The shim writes operation records to its local audit log. Review
                them with{" "}
                <span className="font-mono text-xs">
                  autogpt-shim audit tail
                </span>{" "}
                on your machine. You can stop access with Ctrl+C in the
                daemon&apos;s terminal, or disconnect it with{" "}
                <span className="font-mono text-xs">autogpt-shim revoke</span>.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                This consent applies only to the current chat session. New
                sessions will ask again.
              </Text>
              {consentMutation.isError ? (
                <Text
                  variant="body"
                  role="alert"
                  className="text-sm text-red-700"
                >
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
                expectedFeaturesCoarse,
                expectedFeatures,
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
                expectedFeaturesCoarse,
                expectedFeatures,
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
