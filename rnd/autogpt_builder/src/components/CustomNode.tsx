import React, {
  useState,
  useEffect,
  FC,
  memo,
  useCallback,
  useRef,
} from "react";
import { NodeProps, useReactFlow } from "reactflow";
import "reactflow/dist/style.css";
import "./customnode.css";
import InputModalComponent from "./InputModalComponent";
import OutputModalComponent from "./OutputModalComponent";
import {
  BlockIORootSchema,
  NodeExecutionResult,
} from "@/lib/autogpt-server-api/types";
import { beautifyString, setNestedProperty } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Copy, Trash2 } from "lucide-react";
import { history } from "./history";
import NodeHandle from "./NodeHandle";
import { CustomEdgeData } from "./CustomEdge";
import { NodeGenericInputField } from "./node-input-components";

type ParsedKey = { key: string; index?: number };

export type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: BlockIORootSchema;
  outputSchema: BlockIORootSchema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{
    edge_id: string;
    source: string;
    sourceHandle: string;
    target: string;
    targetHandle: string;
  }>;
  isOutputOpen: boolean;
  status?: NodeExecutionResult["status"];
  output_data?: NodeExecutionResult["output_data"];
  block_id: string;
  backend_id?: string;
  errors?: { [key: string]: string | null };
  setErrors: (errors: { [key: string]: string | null }) => void;
  setIsAnyModalOpen?: (isOpen: boolean) => void;
};

const CustomNode: FC<NodeProps<CustomNodeData>> = ({ data, id }) => {
  const [isOutputOpen, setIsOutputOpen] = useState(data.isOutputOpen || false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [modalValue, setModalValue] = useState<string>("");
  const [isOutputModalOpen, setIsOutputModalOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const { deleteElements } = useReactFlow();

  const outputDataRef = useRef<HTMLDivElement>(null);
  const isInitialSetup = useRef(true);

  useEffect(() => {
    if (data.output_data || data.status) {
      setIsOutputOpen(true);
    }
  }, [data.output_data, data.status]);

  useEffect(() => {
    setIsOutputOpen(data.isOutputOpen);
  }, [data.isOutputOpen]);

  useEffect(() => {
    data.setIsAnyModalOpen?.(isModalOpen || isOutputModalOpen);
  }, [isModalOpen, isOutputModalOpen, data]);

  useEffect(() => {
    isInitialSetup.current = false;
  }, []);

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

    console.log(`Updating hardcoded values for node ${id}:`, newValues);

    if (!isInitialSetup.current) {
      history.push({
        type: "UPDATE_INPUT",
        payload: { nodeId: id, oldValues: data.hardcodedValues, newValues },
        undo: () => data.setHardcodedValues(data.hardcodedValues),
        redo: () => data.setHardcodedValues(newValues),
      });
    }

    data.setHardcodedValues(newValues);
    const errors = data.errors || {};
    // Remove error with the same key
    setNestedProperty(errors, path, null);
    data.setErrors({ ...errors });
  };

  // Helper function to parse keys with array indices
  const parseKeys = (key: string): ParsedKey[] => {
    const regex = /(\w+)|\[(\d+)\]/g;
    const keys: ParsedKey[] = [];
    let match;
    let currentKey: string | null = null;

    while ((match = regex.exec(key)) !== null) {
      if (match[1]) {
        if (currentKey !== null) {
          keys.push({ key: currentKey });
        }
        currentKey = match[1];
      } else if (match[2]) {
        if (currentKey !== null) {
          keys.push({ key: currentKey, index: parseInt(match[2], 10) });
          currentKey = null;
        } else {
          throw new Error("Invalid key format: array index without a key");
        }
      }
    }

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

  const isTextTruncated = (element: HTMLElement | null): boolean => {
    if (!element) return false;
    return (
      element.scrollHeight > element.clientHeight ||
      element.scrollWidth > element.clientWidth
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
    // This is a placeholder function. The actual copy functionality
    // will be implemented by another team member.
    console.log("Copy node:", id);
  }, [id]);

  return (
    <div
      className={`custom-node dark-theme border rounded-xl shadow-md bg-white/[.9] ${data.status?.toLowerCase() ?? ""}`}
      onMouseEnter={handleHovered}
      onMouseLeave={handleMouseLeave}
    >
      <div className="mb-2 p-3 bg-gray-300/[.7] rounded-t-xl">
        <div className="p-3 text-lg font-semibold font-roboto">
          {beautifyString(data.blockType?.replace(/Block$/, "") || data.title)}
        </div>
        <div className="flex gap-[5px] ">
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
      <div className="p-3 flex justify-between items-start gap-2">
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
                          className="mt-1 mb-2"
                          propKey={propKey}
                          propSchema={propSchema}
                          currentValue={getValue(propKey)}
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
        <div className="node-output" onClick={handleOutputClick}>
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
      <div className="flex items-center pl-4 pb-4 mt-2.5">
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
};

export default memo(CustomNode);
