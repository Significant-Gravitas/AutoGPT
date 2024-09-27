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
import { beautifyString, cn, setNestedProperty } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Copy, Trash2 } from "lucide-react";
import { history } from "./history";
import NodeHandle from "./NodeHandle";
import {
  NodeGenericInputField,
  NodeTextBoxInput,
} from "./node-input-components";
import SchemaTooltip from "./SchemaTooltip";
import { getPrimaryCategoryColor } from "@/lib/utils";
import { FlowContext } from "./Flow";
import { Badge } from "./ui/badge";
import DataTable from "./DataTable";
import { IconCoin } from "./ui/icons";
import * as Separator from "@radix-ui/react-separator";

type ParsedKey = { key: string; index?: number };

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

export function CustomNode({ data, id, width, height }: NodeProps<CustomNode>) {
  const [isOutputOpen, setIsOutputOpen] = useState(data.isOutputOpen || false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [inputModalValue, setInputModalValue] = useState<string>("");
  const [isOutputModalOpen, setIsOutputModalOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const { updateNodeData, deleteElements, addNodes, getNode } = useReactFlow<
    CustomNode,
    Edge
  >();
  const isInitialSetup = useRef(true);
  const flowContext = useContext(FlowContext);

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
      case BlockUIType.INPUT:
        // For INPUT blocks, dont include connection handles
        return keys.map(([propKey, propSchema]) => {
          const isRequired = data.inputSchema.required?.includes(propKey);
          const isConnected = isHandleConnected(propKey);
          const isAdvanced = propSchema.advanced;
          return (
            (isRequired || isAdvancedOpen || !isAdvanced) && (
              <div key={propKey}>
                <span className="text-m green -mb-1 text-gray-900">
                  {propSchema.title || beautifyString(propKey)}
                </span>
                <div key={propKey} onMouseOver={() => { }}>
                  {!isConnected && (
                    <NodeGenericInputField
                      className="mb-2 mt-1"
                      propKey={propKey}
                      propSchema={propSchema}
                      currentValue={getValue(propKey)}
                      connections={data.connections}
                      handleInputChange={handleInputChange}
                      handleInputClick={handleInputClick}
                      errors={data.errors ?? {}}
                      displayName={propSchema.title || beautifyString(propKey)}
                    />
                  )}
                </div>
              </div>
            )
          );
        });

      case BlockUIType.NOTE:
        // For NOTE blocks, don't render any input handles
        const [noteKey, noteSchema] = keys[0];
        return (
          <div key={noteKey}>
            <NodeTextBoxInput
              className=""
              selfKey={noteKey}
              schema={noteSchema as BlockIOStringSubSchema}
              value={getValue(noteKey)}
              handleInputChange={handleInputChange}
              handleInputClick={handleInputClick}
              error={data.errors?.[noteKey] ?? ""}
              displayName={noteSchema.title || beautifyString(noteKey)}
            />
          </div>
        );

      case BlockUIType.OUTPUT:
        // For OUTPUT blocks, only show the 'value' property
        return keys.map(([propKey, propSchema]) => {
          const isRequired = data.inputSchema.required?.includes(propKey);
          const isConnected = isHandleConnected(propKey);
          const isAdvanced = propSchema.advanced;
          return (
            (isRequired || isAdvancedOpen || !isAdvanced) && (
              <div key={propKey} onMouseOver={() => { }}>
                {propKey !== "value" ? (
                  <span className="text-m green -mb-1 text-gray-900">
                    {propSchema.title || beautifyString(propKey)}
                  </span>
                ) : (
                  <NodeHandle
                    keyName={propKey}
                    isConnected={isConnected}
                    isRequired={isRequired}
                    schema={propSchema}
                    side="left"
                  />
                )}
                {!isConnected && (
                  <NodeGenericInputField
                    className="mb-2 mt-1"
                    propKey={propKey}
                    propSchema={propSchema}
                    currentValue={getValue(propKey)}
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

      default:
        return keys.map(([propKey, propSchema]) => {
          const isRequired = data.inputSchema.required?.includes(propKey);
          const isConnected = isHandleConnected(propKey);
          const isAdvanced = propSchema.advanced;
          return (
            (isRequired || isAdvancedOpen || isConnected || !isAdvanced) && (
              <div key={propKey} onMouseOver={() => { }}>
                {"credentials_provider" in propSchema ? (
                  <span className="text-m green -mb-1 text-gray-900">
                    Credentials
                  </span>
                ) : (
                  <NodeHandle
                    keyName={propKey}
                    isConnected={isConnected}
                    isRequired={isRequired}
                    schema={propSchema}
                    side="left"
                  />
                )}
                {!isConnected && (
                  <NodeGenericInputField
                    className="mb-2 mt-1"
                    propKey={propKey}
                    propSchema={propSchema}
                    currentValue={getValue(propKey)}
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

    // console.log(`Updating hardcoded values for node ${id}:`, newValues);

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

  // Helper function to parse keys with array indices
  //TODO move to utils
  const parseKeys = (key: string): ParsedKey[] => {
    const splits = key.split(/_@_|_#_|_\$_|\./);
    const keys: ParsedKey[] = [];
    let currentKey: string | null = null;

    splits.forEach((split) => {
      const isInteger = /^\d+$/.test(split);
      if (!isInteger) {
        if (currentKey !== null) {
          keys.push({ key: currentKey });
        }
        currentKey = split;
      } else {
        if (currentKey !== null) {
          keys.push({ key: currentKey, index: parseInt(split, 10) });
          currentKey = null;
        } else {
          throw new Error("Invalid key format: array index without a key");
        }
      }
    });

    if (currentKey !== null) {
      keys.push({ key: currentKey });
    }

    return keys;
  };

  const getValue = (key: string) => {
    const keys = parseKeys(key);
    return keys.reduce((acc, k) => {
      if (acc === undefined) return undefined;
      if (k.index !== undefined) {
        return Array.isArray(acc[k.key]) ? acc[k.key][k.index] : undefined;
      }
      return acc[k.key];
    }, data.hardcodedValues as any);
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
    console.log(`Opening modal for key: ${key}`);
    setActiveKey(key);
    const value = getValue(key);
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

  const handleHovered = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  const deleteNode = useCallback(() => {
    console.log("Deleting node:", id);

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

  const hasConfigErrors =
    data.errors &&
    Object.entries(data.errors).some(([_, value]) => value !== null);
  const outputData = data.executionResults?.at(-1)?.data;
  const hasOutputError =
    typeof outputData === "object" &&
    outputData !== null &&
    "error" in outputData;

  useEffect(() => {
    if (hasConfigErrors) {
      const filteredErrors = Object.fromEntries(
        Object.entries(data.errors || {}).filter(
          ([_, value]) => value !== null,
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
    "border",
    "bg-white/[.9]",
    "shadow-md",
  ]
    .filter(Boolean)
    .join(" ");

  const errorClass =
    hasConfigErrors || hasOutputError ? "border-red-500 border-2" : "";

  const statusClass =
    hasConfigErrors || hasOutputError
      ? "failed"
      : (data.status?.toLowerCase() ?? "");

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

  return (
    <div
      className={`${data.uiType === BlockUIType.NOTE ? "w-[300px]" : "w-[500px]"} ${blockClasses} ${errorClass} ${statusClass} ${data.uiType === BlockUIType.NOTE ? "bg-yellow-100" : "bg-white"}`}
      onMouseEnter={handleHovered}
      onMouseLeave={handleMouseLeave}
      data-id={`custom-node-${id}`}
    >
      {/* Header */}
      <div
        className={`mb-2 flex h-20 border-b-2 ${data.uiType === BlockUIType.NOTE ? "bg-yellow-100" : "bg-white"} rounded-t-xl`}
      >
        {/* Color Stripe */}
        <div
          className={`z-20 flex h-20 min-w-2 flex-shrink-0 rounded-tl-xl border ${getPrimaryCategoryColor(data.categories)}`}
        ></div>

        <div className="flex w-full flex-col">
          <div className="flex flex-row items-center justify-between">
            <div className="font-roboto px-3 text-lg font-semibold">
              {beautifyString(
                data.blockType?.replace(/Block$/, "") || data.title,
              )}
            </div>
          </div>
          {blockCost && (
            <div className="px-3 text-base font-light">
              <span className="ml-auto flex items-center">
                <IconCoin />{" "}
                <span className="m-1 font-medium">{blockCost.cost_amount}</span>{" "}
                per {blockCost.cost_type}
              </span>
            </div>
          )}
        </div>
        <Badge
          variant="outline"
          className={`mr-5 ${getPrimaryCategoryColor(data.categories)} rounded-xl opacity-50`}
        >
          {data.categories[0].category}
        </Badge>
      </div>
      {/* Body */}
      <div className="ml-5">
        {/* Input Handles */}
        {data.uiType !== BlockUIType.NOTE ? (
          <div className="flex items-start justify-between">
            <div>
              {data.inputSchema &&
                generateInputHandles(data.inputSchema, data.uiType)}
            </div>
            {/* <div className="flex-none">
            {data.outputSchema &&
              generateOutputHandles(data.outputSchema, data.uiType)}
          </div> */}
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
            <Separator.Root className="h-[2px] w-full bg-gray-200"></Separator.Root>
            <div className="mt-2.5 flex items-center justify-between pb-4">
              <span className="">Advanced</span>
              <Switch
                onCheckedChange={toggleAdvancedSettings}
                className="mr-5"
              />
            </div>
          </>
        )}
        {/* Output Handles */}
        {data.uiType !== BlockUIType.NOTE ? (
          <>
            <Separator.Root className="h-[2px] w-full bg-gray-200"></Separator.Root>

            <div className="flex items-start justify-end pr-2">
              <div className="flex-none">
                {data.outputSchema &&
                  generateOutputHandles(data.outputSchema, data.uiType)}
              </div>
            </div>
          </>
        ) : (
          <></>
        )}
        {/* Display Outputs  */}
        {isOutputOpen && data.uiType !== BlockUIType.NOTE && (
          <div
            data-id="latest-output"
            className="nodrag m-3 break-words rounded-md border-[1.5px] p-2"
          >
            {(data.executionResults?.length ?? 0) > 0 ? (
              <>
                <DataTable
                  title="Latest Output"
                  truncateLongData
                  data={data.executionResults!.at(-1)?.data || {}}
                />
                <div className="flex justify-end">
                  <Button variant="ghost" onClick={handleOutputClick}>
                    View More
                  </Button>
                </div>
              </>
            ) : (
              <></>
            )}
          </div>
        )}
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
    </div>
    // End Body
  );
}
