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
  Category,
  NodeExecutionResult,
} from "@/lib/autogpt-server-api/types";
import { beautifyString, setNestedProperty } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Copy, Trash2 } from "lucide-react";
import { history } from "./history";
import NodeHandle from "./NodeHandle";
import { NodeGenericInputField } from "./node-input-components";
import SchemaTooltip from "./SchemaTooltip";
import { getPrimaryCategoryColor } from "@/lib/utils";
import { FlowContext } from "./Flow";

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
  title: string;
  description: string;
  categories: Category[];
  inputSchema: BlockIORootSchema;
  outputSchema: BlockIORootSchema;
  hardcodedValues: { [key: string]: any };
  connections: ConnectionData;
  isOutputOpen: boolean;
  status?: NodeExecutionResult["status"];
  output_data?: NodeExecutionResult["output_data"];
  block_id: string;
  backend_id?: string;
  errors?: { [key: string]: string };
  isOutputStatic?: boolean;
};

export type CustomNode = Node<CustomNodeData, "custom">;

export function CustomNode({ data, id, width, height }: NodeProps<CustomNode>) {
  const [isOutputOpen, setIsOutputOpen] = useState(data.isOutputOpen || false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [modalValue, setModalValue] = useState<string>("");
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
    if (data.output_data || data.status) {
      setIsOutputOpen(true);
    }
  }, [data.output_data, data.status]);

  useEffect(() => {
    setIsOutputOpen(data.isOutputOpen);
  }, [data.isOutputOpen]);

  useEffect(() => {
    setIsAnyModalOpen?.(isModalOpen || isOutputModalOpen);
  }, [isModalOpen, isOutputModalOpen, data]);

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

  const hasOptionalFields =
    data.inputSchema &&
    Object.keys(data.inputSchema.properties).some((key) => {
      return !data.inputSchema.required?.includes(key);
    });

  const generateOutputHandles = (schema: BlockIORootSchema) => {
    if (!schema?.properties) return null;
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
    setModalValue(
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
    setModalValue(
      data.output_data
        ? JSON.stringify(data.output_data, null, 2)
        : "[no output (yet)]",
    );
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
      payload: { node: newNode },
      undo: () => deleteElements({ nodes: [{ id: newId }] }),
      redo: () => addNodes(newNode),
    });
  }, [id, data, height, addNodes, deleteElements, getNode, getNextNodeId]);

  const hasConfigErrors =
    data.errors &&
    Object.entries(data.errors).some(([_, value]) => value !== null);
  const outputData = data.output_data;
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

  return (
    <div
      className={`${blockClasses} ${errorClass} ${statusClass}`}
      onMouseEnter={handleHovered}
      onMouseLeave={handleMouseLeave}
      data-id={`custom-node-${id}`}
    >
      <div
        className={`mb-2 p-3 ${getPrimaryCategoryColor(data.categories)} rounded-t-xl`}
      >
        <div className="flex items-center justify-between">
          <div className="font-roboto p-3 text-lg font-semibold">
            {beautifyString(
              data.blockType?.replace(/Block$/, "") || data.title,
            )}
          </div>
          <SchemaTooltip description={data.description} />
        </div>
        <div className="flex gap-[5px]">
          {isHovered && (
            <>
              <Button
                variant="outline"
                size="icon"
                onClick={copyNode}
                title="Copy node"
              >
                <Copy size={18} />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={deleteNode}
                title="Delete node"
              >
                <Trash2 size={18} />
              </Button>
            </>
          )}
        </div>
      </div>
      <div className="flex items-start justify-between gap-2 p-3">
        <div>
          {data.inputSchema &&
            Object.entries(data.inputSchema.properties).map(
              ([propKey, propSchema]) => {
                const isRequired = data.inputSchema.required?.includes(propKey);
                const isConnected = isHandleConnected(propKey);
                return (
                  (isRequired || isAdvancedOpen || isConnected) && (
                    <div key={propKey} onMouseOver={() => {}}>
                      <NodeHandle
                        keyName={propKey}
                        isConnected={isConnected}
                        isRequired={isRequired}
                        schema={propSchema}
                        side="left"
                      />
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
                          displayName={
                            propSchema.title || beautifyString(propKey)
                          }
                        />
                      )}
                    </div>
                  )
                );
              },
            )}
        </div>
        <div className="flex-none">
          {data.outputSchema && generateOutputHandles(data.outputSchema)}
        </div>
      </div>
      {isOutputOpen && (
        <div className="node-output break-words" onClick={handleOutputClick}>
          <p>
            <strong>Status:</strong>{" "}
            {typeof data.status === "object"
              ? JSON.stringify(data.status)
              : data.status || "N/A"}
          </p>
          <p>
            <strong>Output Data:</strong>{" "}
            {(() => {
              const outputText =
                typeof data.output_data === "object"
                  ? JSON.stringify(data.output_data)
                  : data.output_data;

              if (!outputText) return "No output data";

              return outputText.length > 100
                ? `${outputText.slice(0, 100)}... Press To Read More`
                : outputText;
            })()}
          </p>
        </div>
      )}
      <div className="mt-2.5 flex items-center pb-4 pl-4">
        <Switch onCheckedChange={toggleOutput} />
        <span className="m-1 mr-4">Output</span>
        {hasOptionalFields && (
          <>
            <Switch onCheckedChange={toggleAdvancedSettings} />
            <span className="m-1">Advanced</span>
          </>
        )}
      </div>
      <InputModalComponent
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        value={modalValue}
        key={activeKey}
      />
      <OutputModalComponent
        isOpen={isOutputModalOpen}
        onClose={() => setIsOutputModalOpen(false)}
        value={modalValue}
      />
    </div>
  );
}
