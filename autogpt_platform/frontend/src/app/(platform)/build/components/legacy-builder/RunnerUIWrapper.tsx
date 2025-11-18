import { CustomNodeData } from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import {
  BlockUIType,
  CredentialsMetaInput,
  GraphMeta,
} from "@/lib/autogpt-server-api/types";
import { Node } from "@xyflow/react";
import { forwardRef, useImperativeHandle, useMemo, useState } from "react";
import { RunnerInputDialog } from "./RunnerInputUI";
import RunnerOutputUI, { OutputNodeInfo } from "./RunnerOutputUI";

interface RunnerUIWrapperProps {
  graph: GraphMeta;
  nodes: Node<CustomNodeData>[];
  graphExecutionError?: string | null;
  saveAndRun: (
    inputs: Record<string, any>,
    credentialsInputs: Record<string, CredentialsMetaInput>,
  ) => void;
  createRunSchedule: (
    cronExpression: string,
    scheduleName: string,
    inputs: Record<string, any>,
    credentialsInputs: Record<string, CredentialsMetaInput>,
  ) => Promise<void>;
}

export interface RunnerUIWrapperRef {
  openRunInputDialog: () => void;
  openRunnerOutput: () => void;
  runOrOpenInput: (graphOverride?: GraphMeta) => void;
}

const RunnerUIWrapper = forwardRef<RunnerUIWrapperRef, RunnerUIWrapperProps>(
  (
    { graph, nodes, graphExecutionError, saveAndRun, createRunSchedule },
    ref,
  ) => {
    const [isRunInputDialogOpen, setIsRunInputDialogOpen] = useState(false);
    const [isRunnerOutputOpen, setIsRunnerOutputOpen] = useState(false);
    const [graphOverride, setGraphOverride] = useState<GraphMeta | null>(null);
    const graphToUse = graphOverride || graph;

    const graphOutputs = useMemo((): OutputNodeInfo[] => {
      const outputNodes = nodes.filter(
        (node) => node.data.uiType === BlockUIType.OUTPUT,
      );

      return outputNodes.map(
        (node) =>
          ({
            metadata: {
              name: node.data.hardcodedValues.name || "Output",
              description:
                node.data.hardcodedValues.description ||
                "Output from the agent",
            },
            result:
              (node.data.executionResults as any)
                ?.map((result: any) => result?.data?.output)
                .join("\n--\n") || "No output yet",
          }) satisfies OutputNodeInfo,
      );
    }, [nodes]);

    const openRunInputDialog = () => setIsRunInputDialogOpen(true);
    const openRunnerOutput = () => setIsRunnerOutputOpen(true);

    const runOrOpenInput = (graphOverrideParam?: GraphMeta) => {
      if (graphOverrideParam) {
        setGraphOverride(graphOverrideParam);
      }
      const graphToCheck = graphOverrideParam || graph;
      const inputs = graphToCheck.input_schema.properties;
      const credentials = graphToCheck.credentials_input_schema.properties;

      if (
        Object.keys(inputs).length > 0 ||
        Object.keys(credentials).length > 0
      ) {
        openRunInputDialog();
      } else {
        saveAndRun({}, {});
      }
    };

    useImperativeHandle(
      ref,
      () =>
        ({
          openRunInputDialog,
          openRunnerOutput,
          runOrOpenInput,
        }) satisfies RunnerUIWrapperRef,
    );

    return (
      <>
        <RunnerInputDialog
          isOpen={isRunInputDialogOpen}
          doClose={() => {
            setIsRunInputDialogOpen(false);
            setGraphOverride(null);
          }}
          graph={graphToUse}
          doRun={saveAndRun}
          doCreateSchedule={createRunSchedule}
        />
        <RunnerOutputUI
          isOpen={isRunnerOutputOpen}
          doClose={() => setIsRunnerOutputOpen(false)}
          outputs={graphOutputs}
          graphExecutionError={graphExecutionError}
        />
      </>
    );
  },
);

RunnerUIWrapper.displayName = "RunnerUIWrapper";

export default RunnerUIWrapper;
