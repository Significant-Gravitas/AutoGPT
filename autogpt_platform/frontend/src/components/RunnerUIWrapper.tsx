import React, {
  useState,
  useCallback,
  forwardRef,
  useImperativeHandle,
} from "react";
import RunnerInputUI from "./runner-ui/RunnerInputUI";
import RunnerOutputUI from "./runner-ui/RunnerOutputUI";
import { Node } from "@xyflow/react";
import { filterBlocksByType } from "@/lib/utils";
import { BlockIORootSchema, BlockUIType } from "@/lib/autogpt-server-api/types";

interface RunnerUIWrapperProps {
  nodes: Node[];
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  isRunning: boolean;
  requestSaveAndRun: () => void;
}

export interface RunnerUIWrapperRef {
  openRunnerInput: () => void;
  openRunnerOutput: () => void;
  runOrOpenInput: () => void;
}

const RunnerUIWrapper = forwardRef<RunnerUIWrapperRef, RunnerUIWrapperProps>(
  ({ nodes, setNodes, isRunning, requestSaveAndRun }, ref) => {
    const [isRunnerInputOpen, setIsRunnerInputOpen] = useState(false);
    const [isRunnerOutputOpen, setIsRunnerOutputOpen] = useState(false);

    const getBlockInputsAndOutputs = useCallback(() => {
      const inputBlocks = filterBlocksByType(
        nodes,
        (node) => node.data.uiType === BlockUIType.INPUT,
      );

      const outputBlocks = filterBlocksByType(
        nodes,
        (node) => node.data.uiType === BlockUIType.OUTPUT,
      );

      const inputs = inputBlocks.map((node) => ({
        id: node.id,
        type: "input" as const,
        inputSchema: node.data.inputSchema as BlockIORootSchema,
        hardcodedValues: {
          name: (node.data.hardcodedValues as any).name || "",
          description: (node.data.hardcodedValues as any).description || "",
          value: (node.data.hardcodedValues as any).value,
          placeholder_values:
            (node.data.hardcodedValues as any).placeholder_values || [],
          limit_to_placeholder_values:
            (node.data.hardcodedValues as any).limit_to_placeholder_values ||
            false,
        },
      }));

      const outputs = outputBlocks.map((node) => ({
        id: node.id,
        type: "output" as const,
        outputSchema: node.data.outputSchema as BlockIORootSchema,
        hardcodedValues: {
          name: (node.data.hardcodedValues as any).name || "Output",
          description:
            (node.data.hardcodedValues as any).description ||
            "Output from the agent",
          value: (node.data.hardcodedValues as any).value,
        },
        result: (node.data.executionResults as any)?.at(-1)?.data?.output,
      }));

      return { inputs, outputs };
    }, [nodes]);

    const handleInputChange = useCallback(
      (nodeId: string, field: string, value: string) => {
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

    useImperativeHandle(ref, () => ({
      openRunnerInput,
      openRunnerOutput,
      runOrOpenInput,
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
