import React, {
  useState,
  useCallback,
  forwardRef,
  useImperativeHandle,
} from "react";
import RunnerOutputUI, { BlockOutput } from "./runner-ui/RunnerOutputUI";
import RunnerInputUI from "./runner-ui/RunnerInputUI";
import { Node } from "@xyflow/react";
import { filterBlocksByType } from "@/lib/utils";
import {
  BlockIOObjectSubSchema,
  BlockIORootSchema,
  BlockUIType,
} from "@/lib/autogpt-server-api/types";
import { CustomNode } from "./CustomNode";

interface HardcodedValues {
  name: any;
  description: any;
  value: any;
  placeholder_values: any;
}

export interface InputItem {
  id: string;
  type: "input";
  inputSchema: BlockIORootSchema;
  hardcodedValues: HardcodedValues;
}

interface RunnerUIWrapperProps {
  nodes: Node[];
  setNodes: React.Dispatch<React.SetStateAction<CustomNode[]>>;
  setIsScheduling: React.Dispatch<React.SetStateAction<boolean>>;
  isRunning: boolean;
  isScheduling: boolean;
  requestSaveAndRun: () => void;
  scheduleRunner: (cronExpression: string, input: InputItem[]) => Promise<void>;
}

export interface RunnerUIWrapperRef {
  openRunnerInput: () => void;
  openRunnerOutput: () => void;
  runOrOpenInput: () => void;
  collectInputsForScheduling: (cronExpression: string) => void;
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

    const getBlockInputsAndOutputs = useCallback((): {
      inputs: InputItem[];
      outputs: BlockOutput[];
    } => {
      const inputBlocks = filterBlocksByType(
        nodes,
        (node) => node.data.uiType === BlockUIType.INPUT,
      );

      const outputBlocks = filterBlocksByType(
        nodes,
        (node) => node.data.uiType === BlockUIType.OUTPUT,
      );

      const inputs = inputBlocks.map(
        (node) =>
          ({
            id: node.id,
            type: "input" as const,
            inputSchema: (node.data.inputSchema as BlockIOObjectSubSchema)
              .properties.value as BlockIORootSchema,
            hardcodedValues: {
              name: (node.data.hardcodedValues as any).name || "",
              description: (node.data.hardcodedValues as any).description || "",
              value: (node.data.hardcodedValues as any).value,
              placeholder_values:
                (node.data.hardcodedValues as any).placeholder_values || [],
            },
          }) satisfies InputItem,
      );

      const outputs = outputBlocks.map(
        (node) =>
          ({
            metadata: {
              name: (node.data.hardcodedValues as any).name || "Output",
              description:
                (node.data.hardcodedValues as any).description ||
                "Output from the agent",
            },
            result:
              (node.data.executionResults as any)
                ?.map((result: any) => result?.data?.output)
                .join("\n--\n") || "No output yet",
          }) satisfies BlockOutput,
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
                    ...(node.data.hardcodedValues as any),
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
      const { inputs } = getBlockInputsAndOutputs();
      if (inputs.length > 0) {
        openRunnerInput();
      } else {
        requestSaveAndRun();
      }
    };

    const collectInputsForScheduling = (cron_exp: string) => {
      const { inputs } = getBlockInputsAndOutputs();
      setCronExpression(cron_exp);

      if (inputs.length > 0) {
        setScheduledInput(true);
        setIsRunnerInputOpen(true);
      } else {
        scheduleRunner(cron_exp, []);
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
        <RunnerInputUI
          isOpen={isRunnerInputOpen}
          onClose={() => setIsRunnerInputOpen(false)}
          blockInputs={getBlockInputsAndOutputs().inputs}
          onInputChange={handleInputChange}
          onRun={() => {
            setIsRunnerInputOpen(false);
            requestSaveAndRun();
          }}
          scheduledInput={scheduledInput}
          isScheduling={isScheduling}
          onSchedule={async () => {
            setIsScheduling(true);
            await scheduleRunner(
              cronExpression,
              getBlockInputsAndOutputs().inputs,
            );
            setIsScheduling(false);
            setIsRunnerInputOpen(false);
            setScheduledInput(false);
          }}
          isRunning={isRunning}
        />
        <RunnerOutputUI
          isOpen={isRunnerOutputOpen}
          onClose={() => setIsRunnerOutputOpen(false)}
          blockOutputs={getBlockInputsAndOutputs().outputs}
        />
      </>
    );
  },
);

RunnerUIWrapper.displayName = "RunnerUIWrapper";

export default RunnerUIWrapper;
