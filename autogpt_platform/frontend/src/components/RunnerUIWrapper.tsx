import React, {
  useState,
  useCallback,
  forwardRef,
  useImperativeHandle,
} from "react";
import RunnerOutputUI, { OutputNodeInfo } from "./runner-ui/RunnerOutputUI";
import { RunnerInputDialog, InputNodeInfo } from "./runner-ui/RunnerInputUI";
import { Node } from "@xyflow/react";
import { BlockIORootSchema, BlockUIType } from "@/lib/autogpt-server-api/types";
import { CustomNode, CustomNodeData } from "./CustomNode";

interface RunnerUIWrapperProps {
  nodes: Node<CustomNodeData>[];
  setNodes: React.Dispatch<React.SetStateAction<CustomNode[]>>;
  setIsScheduling: React.Dispatch<React.SetStateAction<boolean>>;
  isRunning: boolean;
  isScheduling: boolean;
  requestSaveAndRun: () => void;
  scheduleRunner: (
    cronExpression: string,
    scheduleName: string,
    input: Record<string, any>,
  ) => Promise<void>;
}

export interface RunnerUIWrapperRef {
  openRunnerInput: () => void;
  openRunnerOutput: () => void;
  runOrOpenInput: () => void;
  collectInputsForScheduling: (
    cronExpression: string,
    scheduleName: string,
  ) => void;
}

const RunnerUIWrapper = forwardRef<RunnerUIWrapperRef, RunnerUIWrapperProps>(
  (
    {
      nodes,
      setIsScheduling,
      setNodes,
      isScheduling,
      isRunning,
      requestSaveAndRun,
      scheduleRunner,
    },
    ref,
  ) => {
    const [isRunnerInputOpen, setIsRunnerInputOpen] = useState(false);
    const [isRunnerOutputOpen, setIsRunnerOutputOpen] = useState(false);
    const [scheduledInput, setScheduledInput] = useState(false);
    const [cronExpression, setCronExpression] = useState("");
    const [scheduleName, setScheduleName] = useState("");

    const getGraphInputsAndOutputs = useCallback((): {
      inputs: InputNodeInfo[];
      outputs: OutputNodeInfo[];
    } => {
      const inputNodes = nodes.filter(
        (node) => node.data.uiType === BlockUIType.INPUT,
      );

      const outputNodes = nodes.filter(
        (node) => node.data.uiType === BlockUIType.OUTPUT,
      );

      const inputs = inputNodes.map(
        (node) =>
          ({
            id: node.id,
            inputSchema: node.data.inputSchema.properties
              .value as BlockIORootSchema,
            inputConfig: {
              name: node.data.hardcodedValues.name || "",
              description: node.data.hardcodedValues.description || "",
              defaultValue: node.data.hardcodedValues.value,
              placeholderValues:
                node.data.hardcodedValues.placeholder_values || [],
            },
          }) satisfies InputNodeInfo,
      );

      const outputs = outputNodes.map(
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

      return { inputs, outputs };
    }, [nodes]);

    const handleInputChange = useCallback(
      (nodeId: string, field: string, value: any) => {
        setNodes((nds) =>
          nds.map((node) => {
            if (node.id === nodeId) {
              return {
                ...node,
                data: {
                  ...node.data,
                  hardcodedValues: {
                    ...node.data.hardcodedValues,
                    [field]: value,
                  },
                },
              };
            }
            return node;
          }),
        );
      },
      [setNodes],
    );

    const openRunnerInput = () => setIsRunnerInputOpen(true);
    const openRunnerOutput = () => setIsRunnerOutputOpen(true);

    const runOrOpenInput = () => {
      const { inputs } = getGraphInputsAndOutputs();
      if (inputs.length > 0) {
        openRunnerInput();
      } else {
        requestSaveAndRun();
      }
    };

    const collectInputsForScheduling = (
      cronExpression: string,
      scheduleName: string,
    ) => {
      const { inputs } = getGraphInputsAndOutputs();
      setCronExpression(cronExpression);
      setScheduleName(scheduleName);

      if (inputs.length > 0) {
        setScheduledInput(true);
        setIsRunnerInputOpen(true);
      } else {
        scheduleRunner(cronExpression, scheduleName, {});
      }
    };

    useImperativeHandle(ref, () => ({
      openRunnerInput,
      openRunnerOutput,
      runOrOpenInput,
      collectInputsForScheduling,
    }));

    return (
      <>
        <RunnerInputDialog
          isOpen={isRunnerInputOpen}
          doClose={() => setIsRunnerInputOpen(false)}
          inputs={getGraphInputsAndOutputs().inputs}
          onInputChange={handleInputChange}
          doRun={() => {
            setIsRunnerInputOpen(false);
            requestSaveAndRun();
          }}
          scheduledInput={scheduledInput}
          isScheduling={isScheduling}
          onSchedule={async () => {
            setIsScheduling(true);
            await scheduleRunner(
              cronExpression,
              scheduleName,
              getGraphInputsAndOutputs().inputs.reduce(
                (acc, input) => ({
                  ...acc,
                  [input.inputConfig.name]: input.inputConfig.defaultValue,
                }),
                {},
              ),
            );
            setIsScheduling(false);
            setIsRunnerInputOpen(false);
            setScheduledInput(false);
          }}
          isRunning={isRunning}
        />
        <RunnerOutputUI
          isOpen={isRunnerOutputOpen}
          doClose={() => setIsRunnerOutputOpen(false)}
          outputs={getGraphInputsAndOutputs().outputs}
        />
      </>
    );
  },
);

RunnerUIWrapper.displayName = "RunnerUIWrapper";

export default RunnerUIWrapper;
