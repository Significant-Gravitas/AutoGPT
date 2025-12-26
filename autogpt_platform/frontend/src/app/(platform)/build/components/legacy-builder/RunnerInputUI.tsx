import { useCallback } from "react";

import { AgentRunDraftView } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/agent-run-draft-view";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type {
  CredentialsMetaInput,
  GraphMeta,
} from "@/lib/autogpt-server-api/types";

interface RunInputDialogProps {
  isOpen: boolean;
  doClose: () => void;
  graph: GraphMeta;
  doRun?: (
    inputs: Record<string, any>,
    credentialsInputs: Record<string, CredentialsMetaInput>,
  ) => Promise<void> | void;
  doCreateSchedule?: (
    cronExpression: string,
    scheduleName: string,
    inputs: Record<string, any>,
    credentialsInputs: Record<string, CredentialsMetaInput>,
  ) => Promise<void> | void;
}

export function RunnerInputDialog({
  isOpen,
  doClose,
  graph,
  doRun,
  doCreateSchedule,
}: RunInputDialogProps) {
  const handleRun = useCallback(
    doRun
      ? async (
          inputs: Record<string, any>,
          credentials_inputs: Record<string, CredentialsMetaInput>,
        ) => {
          await doRun(inputs, credentials_inputs);
          doClose();
        }
      : async () => {},
    [doRun, doClose],
  );

  const handleSchedule = useCallback(
    doCreateSchedule
      ? async (
          cronExpression: string,
          scheduleName: string,
          inputs: Record<string, any>,
          credentialsInputs: Record<string, CredentialsMetaInput>,
        ) => {
          await doCreateSchedule(
            cronExpression,
            scheduleName,
            inputs,
            credentialsInputs,
          );
          doClose();
        }
      : async () => {},
    [doCreateSchedule, doClose],
  );

  return (
    <Dialog
      title="Run your agent"
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) doClose();
        },
      }}
      onClose={doClose}
      styling={{
        maxWidth: "56rem",
        width: "90vw",
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col p-10">
          <p className="mt-2 text-sm text-zinc-600">{graph.name}</p>
          <AgentRunDraftView
            className="p-0"
            graph={graph}
            doRun={doRun ? handleRun : undefined}
            onRun={doRun ? undefined : doClose}
            doCreateSchedule={doCreateSchedule ? handleSchedule : undefined}
            onCreateSchedule={doCreateSchedule ? undefined : doClose}
          />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
