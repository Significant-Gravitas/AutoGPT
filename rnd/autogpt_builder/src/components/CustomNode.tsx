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
import { BlockSchema } from "@/lib/types";
import { beautifyString, setNestedProperty } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import NodeHandle from "./NodeHandle";
import NodeInputField from "./NodeInputField";
import { Copy, Trash2 } from "lucide-react";
import { history } from "./history";

export type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: BlockIORootSchema;
  outputSchema: BlockIORootSchema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{
    source: string;
    sourceHandle: string;
    target: string;
    targetHandle: string;
  }>;
  isOutputOpen: boolean;
  status?: string;
  output_data?: any;
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

  const { getNode, setNodes, getEdges, setEdges } = useReactFlow();

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

  const hasOptionalFields = () => {
    return (
      data.inputSchema &&
      Object.keys(data.inputSchema.properties).some((key) => {
        return !data.inputSchema.required?.includes(key);
      })
    );
  };

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

  const handleInputChange = (key: string, value: any) => {
    const keys = key.split(".");
    const newValues = JSON.parse(JSON.stringify(data.hardcodedValues));
    let current = newValues;

    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) current[keys[i]] = {};
      current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = value;

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
    setNestedProperty(errors, key, null);
    data.setErrors({ ...errors });
  };

  const getValue = (key: string) => {
    const keys = key.split(".");
    return keys.reduce(
      (acc, k) => (acc && acc[k] !== undefined ? acc[k] : ""),
      data.hardcodedValues,
    );
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
    console.log("isHovered", isHovered);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    console.log("isHovered", isHovered);
  };

  const deleteNode = useCallback(() => {
    console.log("Deleting node:", id);

    // Get all edges connected to this node
    const connectedEdges = getEdges().filter(
      (edge) => edge.source === id || edge.target === id,
    );

    // For each connected edge, update the connected node's state
    connectedEdges.forEach((edge) => {
      const connectedNodeId = edge.source === id ? edge.target : edge.source;
      const connectedNode = getNode(connectedNodeId);

      if (connectedNode) {
        setNodes((nodes) =>
          nodes.map((node) => {
            if (node.id === connectedNodeId) {
              // Update the node's data to reflect the disconnection
              const updatedConnections = node.data.connections.filter(
                (conn) => !(conn.source === id || conn.target === id),
              );
              return {
                ...node,
                data: {
                  ...node.data,
                  connections: updatedConnections,
                },
              };
            }
            return node;
          }),
        );
      }
    });

    // Remove the node and its connected edges
    setNodes((nodes) => nodes.filter((node) => node.id !== id));
    setEdges((edges) =>
      edges.filter((edge) => edge.source !== id && edge.target !== id),
    );
  }, [id, setNodes, setEdges, getNode, getEdges]);

  const copyNode = useCallback(() => {
    // This is a placeholder function. The actual copy functionality
    // will be implemented by another team member.
    console.log("Copy node:", id);
  }, [id]);

  return (
    <div
      className={`custom-node dark-theme ${data.status?.toLowerCase() ?? ""}`}
      onMouseEnter={handleHovered}
      onMouseLeave={handleMouseLeave}
    >
      <div className="mb-2">
        <div className="text-lg font-bold">
          {beautifyString(data.blockType?.replace(/Block$/, "") || data.title)}
        </div>
        <div className="node-actions">
          {isHovered && (
            <>
              <button
                className="node-action-button"
                onClick={copyNode}
                title="Copy node"
              >
                <Copy size={18} />
              </button>
              <button
                className="node-action-button"
                onClick={deleteNode}
                title="Delete node"
              >
                <Trash2 size={18} />
              </button>
            </>
          )}
        </div>
      </div>
      <div className="node-content">
        <div>
          {data.inputSchema &&
            Object.entries(data.inputSchema.properties).map(([key, schema]) => {
              const isRequired = data.inputSchema.required?.includes(key);
              return (
                (isRequired || isAdvancedOpen) && (
                  <div key={key} onMouseOver={() => {}}>
                    <NodeHandle
                      keyName={key}
                      isConnected={isHandleConnected(key)}
                      isRequired={isRequired}
                      schema={schema}
                      side="left"
                    />
                    {!isHandleConnected(key) && (
                      <NodeInputField
                        keyName={key}
                        schema={schema}
                        value={getValue(key)}
                        handleInputClick={handleInputClick}
                        handleInputChange={handleInputChange}
                        errors={data.errors?.[key]}
                      />
                    )}
                  </div>
                )
              );
            })}
        </div>
        <div>
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
      <div className="flex items-center mt-2.5">
        <Switch onCheckedChange={toggleOutput} className="custom-switch" />
        <span className="m-1 mr-4">Output</span>
        {hasOptionalFields() && (
          <>
            <Switch
              onCheckedChange={toggleAdvancedSettings}
              className="custom-switch"
            />
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
