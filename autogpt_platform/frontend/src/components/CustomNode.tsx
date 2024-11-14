import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useContext,
} from "react";
import { NodeProps, useReactFlow, Node, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./customnode.css";
import InputModalComponent from "./InputModalComponent";
import OutputModalComponent from "./OutputModalComponent";
import {
  BlockIORootSchema,
  BlockIOStringSubSchema,
  Category,
  NodeExecutionResult,
  BlockUIType,
  BlockCost,
} from "@/lib/autogpt-server-api/types";
import {
  beautifyString,
  cn,
  getValue,
  hasNonNullNonObjectValue,
  parseKeys,
  setNestedProperty,
} from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { history } from "./history";
import NodeHandle from "./NodeHandle";
import {
  NodeGenericInputField,
  NodeTextBoxInput,
} from "./node-input-components";
import { getPrimaryCategoryColor } from "@/lib/utils";
import { FlowContext } from "./Flow";
import { Badge } from "./ui/badge";
import NodeOutputs from "./NodeOutputs";
import { IconCoin } from "./ui/icons";
import * as Separator from "@radix-ui/react-separator";
import * as ContextMenu from "@radix-ui/react-context-menu";
import {
  DotsVerticalIcon,
  TrashIcon,
  CopyIcon,
  ExitIcon,
} from "@radix-ui/react-icons";

export type ConnectionData = Array<{
  edge_id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}>;

export type CustomNodeData = {
  blockType: string;
  blockCosts: BlockCost[];
  title: string;
  description: string;
  categories: Category[];
  inputSchema: BlockIORootSchema;
  outputSchema: BlockIORootSchema;
  hardcodedValues: { [key: string]: any };
  connections: ConnectionData;
  isOutputOpen: boolean;
  status?: NodeExecutionResult["status"];
  /** executionResults contains outputs across multiple executions
   * with the last element being the most recent output */
  executionResults?: {
    execId: string;
    data: NodeExecutionResult["output_data"];
  }[];
  block_id: string;
  backend_id?: string;
  errors?: { [key: string]: string };
  isOutputStatic?: boolean;
  uiType: BlockUIType;
};

export type CustomNode = Node<CustomNodeData, "custom">;

export function CustomNode({
  data,
  id,
  width,
  height,
  selected,
}: NodeProps<CustomNode>) {
  const [isOutputOpen, setIsOutputOpen] = useState(data.isOutputOpen || false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [inputModalValue, setInputModalValue] = useState<string>("");
  const [isOutputModalOpen, setIsOutputModalOpen] = useState(false);
  const { updateNodeData, deleteElements, addNodes, getNode } = useReactFlow<
    CustomNode,
    Edge
  >();
  const isInitialSetup = useRef(true);
  const flowContext = useContext(FlowContext);
  let nodeFlowId = "";

  if (data.uiType === BlockUIType.AGENT) {
    // Display the graph's schema instead AgentExecutorBlock's schema.
    data.inputSchema = data.hardcodedValues?.input_schema || {};
    data.outputSchema = data.hardcodedValues?.output_schema || {};
    nodeFlowId = data.hardcodedValues?.graph_id || nodeFlowId;
  }

  if (!flowContext) {
    throw new Error("FlowContext consumer must be inside FlowEditor component");
  }

  const { setIsAnyModalOpen, getNextNodeId } = flowContext;

  useEffect(() => {
    if (data.executionResults || data.status) {
      setIsOutputOpen(true);
    }
  }, [data.executionResults, data.status]);

  useEffect(() => {
    setIsOutputOpen(data.isOutputOpen);
  }, [data.isOutputOpen]);

  useEffect(() => {
    setIsAnyModalOpen?.(isModalOpen || isOutputModalOpen);
  }, [isModalOpen, isOutputModalOpen, data, setIsAnyModalOpen]);

  useEffect(() => {
    isInitialSetup.current = false;
  }, []);

  const setHardcodedValues = (values: any) => {
    updateNodeData(id, { hardcodedValues: values });
  };

  const setErrors = (errors: { [key: string]: string }) => {
    updateNodeData(id, { errors });
  };

  const toggleOutput = (checked: boolean) => {
    setIsOutputOpen(checked);
  };

  const toggleAdvancedSettings = (checked: boolean) => {
    setIsAdvancedOpen(checked);
  };

  const generateOutputHandles = (
    schema: BlockIORootSchema,
    nodeType: BlockUIType,
  ) => {
    if (
      !schema?.properties ||
      nodeType === BlockUIType.OUTPUT ||
      nodeType === BlockUIType.NOTE
    )
      return null;
    const keys = Object.keys(schema.properties);
    return keys.map((key) => (
      <div key={key}>
        <NodeHandle
          keyName={key}
          isConnected={isHandleConnected(key)}
          schema={schema.properties[key]}
          side="right"
        />
      </div>
    ));
  };

  const generateInputHandles = (
    schema: BlockIORootSchema,
    nodeType: BlockUIType,
  ) => {
    if (!schema?.properties) return null;
    let keys = Object.entries(schema.properties);
    switch (nodeType) {
      case BlockUIType.NOTE:
        // For NOTE blocks, don't render any input handles
        const [noteKey, noteSchema] = keys[0];
        return (
          <div key={noteKey}>
            <NodeTextBoxInput
              className=""
              selfKey={noteKey}
              schema={noteSchema as BlockIOStringSubSchema}
              value={getValue(noteKey, data.hardcodedValues)}
              handleInputChange={handleInputChange}
              handleInputClick={handleInputClick}
              error={data.errors?.[noteKey] ?? ""}
              displayName={noteSchema.title || beautifyString(noteKey)}
            />
          </div>
        );

      default:
        const getInputPropKey = (key: string) =>
          nodeType == BlockUIType.AGENT ? `data.${key}` : key;

        return keys.map(([propKey, propSchema]) => {
          const isRequired = data.inputSchema.required?.includes(propKey);
          const isConnected = isHandleConnected(propKey);
          const isAdvanced = propSchema.advanced;
          const isConnectable =
            // No input connection handles for credentials
            propKey !== "credentials" &&
            // No input connection handles on INPUT blocks
            nodeType !== BlockUIType.INPUT &&
            // For OUTPUT blocks, only show the 'value' (hides 'name') input connection handle
            !(nodeType == BlockUIType.OUTPUT && propKey == "name");
          return (
            (isRequired || isAdvancedOpen || isConnected || !isAdvanced) && (
              <div key={propKey} data-id={`input-handle-${propKey}`}>
                {isConnectable ? (
                  <NodeHandle
                    keyName={propKey}
                    isConnected={isConnected}
                    isRequired={isRequired}
                    schema={propSchema}
                    side="left"
                  />
                ) : (
                  propKey != "credentials" && (
                    <span
                      className="text-m green mb-0 text-gray-900"
                      title={propSchema.description}
                    >
                      {propSchema.title || beautifyString(propKey)}
                    </span>
                  )
                )}
                {!isConnected && (
                  <NodeGenericInputField
                    nodeId={id}
                    propKey={getInputPropKey(propKey)}
                    propSchema={propSchema}
                    currentValue={getValue(
                      getInputPropKey(propKey),
                      data.hardcodedValues,
                    )}
                    connections={data.connections}
                    handleInputChange={handleInputChange}
                    handleInputClick={handleInputClick}
                    errors={data.errors ?? {}}
                    displayName={propSchema.title || beautifyString(propKey)}
                  />
                )}
              </div>
            )
          );
        });
    }
  };
  const handleInputChange = (path: string, value: any) => {
    const keys = parseKeys(path);
    const newValues = JSON.parse(JSON.stringify(data.hardcodedValues));
    let current = newValues;

    for (let i = 0; i < keys.length - 1; i++) {
      const { key: currentKey, index } = keys[i];
      if (index !== undefined) {
        if (!current[currentKey]) current[currentKey] = [];
        if (!current[currentKey][index]) current[currentKey][index] = {};
        current = current[currentKey][index];
      } else {
        if (!current[currentKey]) current[currentKey] = {};
        current = current[currentKey];
      }
    }

    const lastKey = keys[keys.length - 1];
    if (lastKey.index !== undefined) {
      if (!current[lastKey.key]) current[lastKey.key] = [];
      current[lastKey.key][lastKey.index] = value;
    } else {
      current[lastKey.key] = value;
    }

    if (!isInitialSetup.current) {
      history.push({
        type: "UPDATE_INPUT",
        payload: { nodeId: id, oldValues: data.hardcodedValues, newValues },
        undo: () => setHardcodedValues(data.hardcodedValues),
        redo: () => setHardcodedValues(newValues),
      });
    }

    setHardcodedValues(newValues);
    const errors = data.errors || {};
    // Remove error with the same key
    setNestedProperty(errors, path, null);
    setErrors({ ...errors });
  };

  const isHandleConnected = (key: string) => {
    return (
      data.connections &&
      data.connections.some((conn: any) => {
        if (typeof conn === "string") {
          const [source, target] = conn.split(" -> ");
          return (
            (target.includes(key) && target.includes(data.title)) ||
            (source.includes(key) && source.includes(data.title))
          );
        }
        return (
          (conn.target === id && conn.targetHandle === key) ||
          (conn.source === id && conn.sourceHandle === key)
        );
      })
    );
  };

  const handleInputClick = (key: string) => {
    console.debug(`Opening modal for key: ${key}`);
    setActiveKey(key);
    const value = getValue(key, data.hardcodedValues);
    setInputModalValue(
      typeof value === "object" ? JSON.stringify(value, null, 2) : value,
    );
    setIsModalOpen(true);
  };

  const handleModalSave = (value: string) => {
    if (activeKey) {
      try {
        const parsedValue = JSON.parse(value);
        handleInputChange(activeKey, parsedValue);
      } catch (error) {
        handleInputChange(activeKey, value);
      }
    }
    setIsModalOpen(false);
    setActiveKey(null);
  };

  const handleOutputClick = () => {
    setIsOutputModalOpen(true);
  };

  const deleteNode = useCallback(() => {
    console.debug("Deleting node:", id);

    // Remove the node
    deleteElements({ nodes: [{ id }] });
  }, [id, deleteElements]);

  const copyNode = useCallback(() => {
    const newId = getNextNodeId();
    const currentNode = getNode(id);

    if (!currentNode) {
      console.error("Cannot copy node: current node not found");
      return;
    }

    const verticalOffset = height ?? 100;

    const newNode: CustomNode = {
      id: newId,
      type: currentNode.type,
      position: {
        x: currentNode.position.x,
        y: currentNode.position.y - verticalOffset - 20,
      },
      data: {
        ...data,
        title: `${data.title} (Copy)`,
        block_id: data.block_id,
        connections: [],
        isOutputOpen: false,
      },
    };

    addNodes(newNode);

    history.push({
      type: "ADD_NODE",
      payload: { node: { ...newNode, ...newNode.data } as CustomNodeData },
      undo: () => deleteElements({ nodes: [{ id: newId }] }),
      redo: () => addNodes(newNode),
    });
  }, [id, data, height, addNodes, deleteElements, getNode, getNextNodeId]);

  const hasConfigErrors = data.errors && hasNonNullNonObjectValue(data.errors);
  const outputData = data.executionResults?.at(-1)?.data;
  const hasOutputError =
    typeof outputData === "object" &&
    outputData !== null &&
    "error" in outputData;

  useEffect(() => {
    if (hasConfigErrors) {
      const filteredErrors = Object.fromEntries(
        Object.entries(data.errors || {}).filter(([, value]) =>
          hasNonNullNonObjectValue(value),
        ),
      );
      console.error(
        "Block configuration errors for",
        data.title,
        ":",
        filteredErrors,
      );
    }
    if (hasOutputError) {
      console.error(
        "Block output contains error for",
        data.title,
        ":",
        outputData.error,
      );
    }
  }, [hasConfigErrors, hasOutputError, data.errors, outputData, data.title]);

  const blockClasses = [
    "custom-node",
    "dark-theme",
    "rounded-xl",
    "bg-white/[.9]",
    "border border-gray-300",
    data.uiType === BlockUIType.NOTE ? "w-[300px]" : "w-[500px]",
    data.uiType === BlockUIType.NOTE ? "bg-yellow-100" : "bg-white",
    selected ? "shadow-2xl" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const errorClass =
    hasConfigErrors || hasOutputError ? "border-red-200 border-2" : "";

  const statusClass = (() => {
    if (hasConfigErrors || hasOutputError) return "border-red-200 border-4";
    switch (data.status?.toLowerCase()) {
      case "completed":
        return "border-green-200 border-4";
      case "running":
        return "border-yellow-200 border-4";
      case "failed":
        return "border-red-200 border-4";
      case "incomplete":
        return "border-purple-200 border-4";
      case "queued":
        return "border-cyan-200 border-4";
      default:
        return "";
    }
  })();

  const statusBackgroundClass = (() => {
    if (hasConfigErrors || hasOutputError) return "bg-red-200";
    switch (data.status?.toLowerCase()) {
      case "completed":
        return "bg-green-200";
      case "running":
        return "bg-yellow-200";
      case "failed":
        return "bg-red-200";
      case "incomplete":
        return "bg-purple-200";
      case "queued":
        return "bg-cyan-200";
      default:
        return "";
    }
  })();

  const hasAdvancedFields =
    data.inputSchema &&
    Object.entries(data.inputSchema.properties).some(([key, value]) => {
      return (
        value.advanced === true && !data.inputSchema.required?.includes(key)
      );
    });

  const inputValues = data.hardcodedValues;
  const blockCost =
    data.blockCosts &&
    data.blockCosts.find((cost) =>
      Object.entries(cost.cost_filter).every(
        // Undefined, null, or empty values are considered equal
        ([key, value]) =>
          value === inputValues[key] || (!value && !inputValues[key]),
      ),
    );

  const LineSeparator = () => (
    <div className="bg-white pt-6">
      <Separator.Root className="h-[1px] w-full bg-gray-300"></Separator.Root>
    </div>
  );

  const ContextMenuContent = () => (
    <ContextMenu.Content className="z-10 rounded-xl border bg-white p-1 shadow-md">
      <ContextMenu.Item
        onSelect={copyNode}
        className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100"
      >
        <CopyIcon className="mr-2 h-5 w-5" />
        <span>Copy</span>
      </ContextMenu.Item>
      {nodeFlowId && (
        <ContextMenu.Item
          onSelect={() => window.open(`/build?flowID=${nodeFlowId}`)}
          className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100"
        >
          <ExitIcon className="mr-2 h-5 w-5" />
          <span>Open agent</span>
        </ContextMenu.Item>
      )}
      <ContextMenu.Separator className="my-1 h-px bg-gray-300" />
      <ContextMenu.Item
        onSelect={deleteNode}
        className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-gray-100"
      >
        <TrashIcon className="mr-2 h-5 w-5 text-red-500" />
        <span>Delete</span>
      </ContextMenu.Item>
    </ContextMenu.Content>
  );

  const onContextButtonTrigger = (e: React.MouseEvent) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const event = new MouseEvent("contextmenu", {
      bubbles: true,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
    });
    e.currentTarget.dispatchEvent(event);
  };

  const stripeColor = getPrimaryCategoryColor(data.categories);

  const nodeContent = () => (
    <div
      className={`${blockClasses} ${errorClass} ${statusClass}`}
      data-id={`custom-node-${id}`}
      z-index={1}
    >
      {/* Header */}
      <div
        className={`flex h-24 border-b border-gray-300 ${data.uiType === BlockUIType.NOTE ? "bg-yellow-100" : "bg-white"} items-center rounded-t-xl`}
      >
        {/* Color Stripe */}
        <div className={`-ml-px h-full w-3 rounded-tl-xl ${stripeColor}`}></div>

        <div className="flex w-full flex-col">
          <div className="flex flex-row items-center justify-between">
            <div className="font-roboto flex items-center px-3 text-lg font-semibold">
              {beautifyString(
                data.blockType?.replace(/Block$/, "") || data.title,
              )}
              <div className="px-2 text-xs text-gray-500">
                #{id.split("-")[0]}
              </div>
            </div>
          </div>
          {blockCost && (
            <div className="px-3 text-base font-light">
              <span className="ml-auto flex items-center">
                <IconCoin />{" "}
                <span className="m-1 font-medium">{blockCost.cost_amount}</span>{" "}
                credits/{blockCost.cost_type}
              </span>
            </div>
          )}
        </div>
        {data.categories.map((category) => (
          <Badge
            key={category.category}
            variant="outline"
            className={`mr-5 ${getPrimaryCategoryColor([category])} whitespace-nowrap rounded-xl border border-gray-300 opacity-50`}
          >
            {beautifyString(category.category.toLowerCase())}
          </Badge>
        ))}
        <button
          aria-label="Options"
          className="mr-2 cursor-pointer rounded-full border-none bg-transparent p-1 hover:bg-gray-100"
          onClick={onContextButtonTrigger}
        >
          <DotsVerticalIcon className="h-5 w-5" />
        </button>

        <ContextMenuContent />
      </div>
      {/* Body */}
      <div className="ml-5 mt-6 rounded-b-xl">
        {/* Input Handles */}
        {data.uiType !== BlockUIType.NOTE ? (
          <div
            className="flex w-fit items-start justify-between"
            data-id="input-handles"
          >
            <div>
              {data.inputSchema &&
                generateInputHandles(data.inputSchema, data.uiType)}
            </div>
          </div>
        ) : (
          <div>
            {data.inputSchema &&
              generateInputHandles(data.inputSchema, data.uiType)}
          </div>
        )}

        {/* Advanced Settings */}
        {data.uiType !== BlockUIType.NOTE && hasAdvancedFields && (
          <>
            <LineSeparator />
            <div className="flex items-center justify-between pt-6">
              Advanced
              <Switch
                onCheckedChange={toggleAdvancedSettings}
                checked={isAdvancedOpen}
                className="mr-5"
              />
            </div>
          </>
        )}
        {/* Output Handles */}
        {data.uiType !== BlockUIType.NOTE && (
          <>
            <LineSeparator />
            <div className="flex items-start justify-end rounded-b-xl pb-2 pr-2 pt-6">
              <div className="flex-none">
                {data.outputSchema &&
                  generateOutputHandles(data.outputSchema, data.uiType)}
              </div>
            </div>
          </>
        )}
      </div>
      {/* End Body */}
      {/* Footer */}
      <div className="flex rounded-b-xl">
        {/* Display Outputs  */}
        {isOutputOpen && data.uiType !== BlockUIType.NOTE && (
          <div
            data-id="latest-output"
            className={cn(
              "nodrag w-full overflow-hidden break-words",
              statusBackgroundClass,
            )}
          >
            {(data.executionResults?.length ?? 0) > 0 ? (
              <div className="mt-0 rounded-b-xl bg-gray-50">
                <LineSeparator />
                <NodeOutputs
                  title="Latest Output"
                  truncateLongData
                  data={data.executionResults!.at(-1)?.data || {}}
                />
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    onClick={handleOutputClick}
                    className="border border-gray-300"
                  >
                    View More
                  </Button>
                </div>
              </div>
            ) : (
              <div className="mt-0 min-h-4 rounded-b-xl bg-white"></div>
            )}
            <div
              className={cn(
                "flex min-h-12 items-center justify-end",
                statusBackgroundClass,
              )}
            >
              <Badge
                variant="default"
                data-id={`badge-${id}-${data.status}`}
                className={cn(
                  "mr-4 flex min-w-[114px] items-center justify-center rounded-3xl text-center text-xs font-semibold",
                  hasConfigErrors || hasOutputError
                    ? "border-red-600 bg-red-600 text-white"
                    : {
                        "border-green-600 bg-green-600 text-white":
                          data.status === "COMPLETED",
                        "border-yellow-600 bg-yellow-600 text-white":
                          data.status === "RUNNING",
                        "border-red-600 bg-red-600 text-white":
                          data.status === "FAILED",
                        "border-blue-600 bg-blue-600 text-white":
                          data.status === "QUEUED",
                        "border-gray-600 bg-gray-600 font-black":
                          data.status === "INCOMPLETE",
                      },
                )}
              >
                {hasConfigErrors || hasOutputError
                  ? "Error"
                  : data.status
                    ? beautifyString(data.status)
                    : "Not Run"}
              </Badge>
            </div>
          </div>
        )}
      </div>
      <InputModalComponent
        title={activeKey ? `Enter ${beautifyString(activeKey)}` : undefined}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        defaultValue={inputModalValue}
        key={activeKey}
      />
      <OutputModalComponent
        isOpen={isOutputModalOpen}
        onClose={() => setIsOutputModalOpen(false)}
        executionResults={data.executionResults?.toReversed() || []}
      />
    </div>
  );

  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger>{nodeContent()}</ContextMenu.Trigger>
    </ContextMenu.Root>
  );
}
