import React, { useCallback } from "react";

import type {
  CredentialsMetaInput,
  GraphMeta,
} from "@/lib/autogpt-server-api/types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/__legacy__/ui/dialog";
import { AgentRunDraftView } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/agent-run-draft-view";

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
    <Dialog open={isOpen} onOpenChange={doClose}>
      <DialogContent className="flex w-[90vw] max-w-4xl flex-col p-10">
        <DialogHeader>
          <DialogTitle className="text-2xl">Run your agent</DialogTitle>
          <DialogDescription className="mt-2">{graph.name}</DialogDescription>
        </DialogHeader>
        <AgentRunDraftView
          className="p-0"
          graph={graph}
          doRun={doRun ? handleRun : undefined}
          onRun={doRun ? undefined : doClose}
          doCreateSchedule={doCreateSchedule ? handleSchedule : undefined}
          onCreateSchedule={doCreateSchedule ? undefined : doClose}
          runCount={0}
        />
      </DialogContent>
    </Dialog>
  );
}
