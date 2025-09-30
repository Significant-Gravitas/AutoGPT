import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useContext,
} from "react";
import Link from "next/link";
import { NodeProps, useReactFlow, Node as XYNode, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./customnode.css";
import InputModalComponent from "../InputModalComponent";
import OutputModalComponent from "../OutputModalComponent";
import {
  BlockIORootSchema,
  BlockIOSubSchema,
  BlockIOStringSubSchema,
  Category,
  NodeExecutionResult,
  BlockUIType,
  BlockCost,
} from "@/lib/autogpt-server-api";
import {
  beautifyString,
  cn,
  getValue,
  hasNonNullNonObjectValue,
  isObject,
  parseKeys,
  setNestedProperty,
} from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import { TextRenderer } from "@/components/__legacy__/ui/render";
import { history } from "../history";
import NodeHandle from "../NodeHandle";
import { NodeGenericInputField, NodeTextBoxInput } from "../NodeInputs";
import { getPrimaryCategoryColor } from "@/lib/utils";
import { BuilderContext } from "../Flow/Flow";
import { Badge } from "../../../../../../components/__legacy__/ui/badge";
import NodeOutputs from "../NodeOutputs";
import { IconCoin } from "../../../../../../components/__legacy__/ui/icons";
import * as Separator from "@radix-ui/react-separator";
import * as ContextMenu from "@radix-ui/react-context-menu";
import {
  Alert,
  AlertDescription,
} from "../../../../../../components/molecules/Alert/Alert";
import {
  DotsVerticalIcon,
  TrashIcon,
  CopyIcon,
  ExitIcon,
  Pencil1Icon,
} from "@radix-ui/react-icons";
import { InfoIcon, Key } from "@phosphor-icons/react";
import useCredits from "@/hooks/useCredits";
import { getV1GetAyrshareSsoUrl } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Input } from "@/components/__legacy__/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { Switch } from "@/components/atoms/Switch/Switch";

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
    status: NodeExecutionResult["status"];
  }[];
  block_id: string;
  backend_id?: string;
  errors?: { [key: string]: string };
  isOutputStatic?: boolean;
  uiType: BlockUIType;
  metadata?: { [key: string]: any };
};

export type CustomNode = XYNode<CustomNodeData, "custom">;

export const CustomNode = React.memo(
  function CustomNode({ data, id, height, selected }: NodeProps<CustomNode>) {
    const [isOutputOpen, setIsOutputOpen] = useState(
      data.isOutputOpen || false,
    );
    const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [activeKey, setActiveKey] = useState<string | null>(null);
    const [inputModalValue, setInputModalValue] = useState<string>("");
    const [isOutputModalOpen, setIsOutputModalOpen] = useState(false);
    const [isEditingTitle, setIsEditingTitle] = useState(false);
    const [customTitle, setCustomTitle] = useState(
      data.metadata?.customized_name || "",
    );
    const [isTitleHovered, setIsTitleHovered] = useState(false);
    const titleInputRef = useRef<HTMLInputElement>(null);
    const { updateNodeData, deleteElements, addNodes, getNode } = useReactFlow<
      CustomNode,
      Edge
    >();
    const isInitialSetup = useRef(true);
    const builderContext = useContext(BuilderContext);
    const { formatCredits } = useCredits();
    const [isLoading, setIsLoading] = useState(false);

    let subGraphID = "";

    if (data.uiType === BlockUIType.AGENT) {
      // Display the graph's schema instead AgentExecutorBlock's schema.
      data.inputSchema = data.hardcodedValues?.input_schema || {};
      data.outputSchema = data.hardcodedValues?.output_schema || {};
      subGraphID = data.hardcodedValues?.graph_id || subGraphID;
    }

    if (!builderContext) {
      throw new Error(
        "BuilderContext consumer must be inside FlowEditor component",
      );
    }

    const { libraryAgent, setIsAnyModalOpen, getNextNodeId } = builderContext;

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

    const fillDefaults = useCallback((obj: any, schema: any) => {
      // Iterate over the schema properties
      for (const key in schema.properties) {
        if (schema.properties.hasOwnProperty(key)) {
          const propertySchema = schema.properties[key];

          // If the property is not in the object, initialize it with the default value
          if (!obj.hasOwnProperty(key)) {
            if (propertySchema.default !== undefined) {
              obj[key] = propertySchema.default;
            } else if (propertySchema.type === "object") {
              // Recursively fill defaults for nested objects
              obj[key] = fillDefaults({}, propertySchema);
            } else if (propertySchema.type === "array") {
              // Recursively fill defaults for arrays
              obj[key] = fillDefaults([], propertySchema);
            }
          } else {
            // If the property exists, recursively fill defaults for nested objects/arrays
            if (propertySchema.type === "object") {
              obj[key] = fillDefaults(obj[key], propertySchema);
            } else if (propertySchema.type === "array") {
              obj[key] = fillDefaults(obj[key], propertySchema);
            }
          }
        }
      }

      return obj;
    }, []);

    const setHardcodedValues = useCallback(
      (values: any) => {
        updateNodeData(id, { hardcodedValues: values });
      },
      [id, updateNodeData],
    );

    const handleTitleEdit = useCallback(() => {
      setIsEditingTitle(true);
      setTimeout(() => {
        titleInputRef.current?.focus();
        titleInputRef.current?.select();
      }, 0);
    }, []);

    const handleTitleSave = useCallback(() => {
      setIsEditingTitle(false);
      const newMetadata = {
        ...data.metadata,
        customized_name: customTitle.trim() || undefined,
      };
      updateNodeData(id, { metadata: newMetadata });
    }, [customTitle, data.metadata, id, updateNodeData]);

    const handleTitleKeyDown = useCallback(
      (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
          handleTitleSave();
        } else if (e.key === "Escape") {
          setCustomTitle(data.metadata?.customized_name || "");
          setIsEditingTitle(false);
        }
      },
      [handleTitleSave, data.metadata],
    );

    const displayTitle =
      customTitle ||
      beautifyString(data.blockType?.replace(/Block$/, "") || data.title);

    useEffect(() => {
      isInitialSetup.current = false;
      if (data.uiType === BlockUIType.AGENT) {
        setHardcodedValues({
          ...data.hardcodedValues,
          inputs: fillDefaults(
            data.hardcodedValues.inputs ?? {},
            data.inputSchema,
          ),
        });
      } else {
        setHardcodedValues(
          fillDefaults(data.hardcodedValues, data.inputSchema),
        );
      }
    }, []);

    const setErrors = useCallback(
      (errors: { [key: string]: string }) => {
        updateNodeData(id, { errors });
      },
      [id, updateNodeData],
    );

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

      const renderHandles = (
        propSchema: { [key: string]: BlockIOSubSchema },
        keyPrefix = "",
        titlePrefix = "",
      ) => {
        return Object.keys(propSchema).map((propKey) => {
          const fieldSchema = propSchema[propKey];
          const fieldTitle =
            titlePrefix + (fieldSchema.title || beautifyString(propKey));

          return (
            <div key={propKey}>
              <NodeHandle
                title={fieldTitle}
                keyName={`${keyPrefix}${propKey}`}
                isConnected={isOutputHandleConnected(propKey)}
                schema={fieldSchema}
                side="right"
              />
              {"properties" in fieldSchema &&
                renderHandles(
                  fieldSchema.properties,
                  `${keyPrefix}${propKey}_#_`,
                  `${fieldTitle}.`,
                )}
            </div>
          );
        });
      };

      return renderHandles(schema.properties);
    };

    const generateAyrshareSSOHandles = () => {
      const handleSSOLogin = async () => {
        setIsLoading(true);
        try {
          const { data, status } = await getV1GetAyrshareSsoUrl();
          if (status !== 200) {
            throw new Error(data.detail);
          }
          const popup = window.open(data.sso_url, "_blank", "popup=true");
          if (!popup) {
            throw new Error(
              "Please allow popups for this site to be able to login with Ayrshare",
            );
          }
        } catch (error) {
          toast({
            title: "Error",
            description: `Error getting SSO URL: ${error}`,
            variant: "destructive",
          });
        } finally {
          setIsLoading(false);
        }
      };

      return (
        <div className="flex flex-col gap-2">
          <Button type="button" onClick={handleSSOLogin} disabled={isLoading}>
            {isLoading ? (
              "Loading..."
            ) : (
              <>
                <Key className="mr-2 h-4 w-4" />
                Connect Social Media Accounts
              </>
            )}
          </Button>
          <NodeHandle
            title="SSO Token"
            keyName="sso_token"
            isConnected={false}
            schema={{ type: "string" }}
            side="right"
          />
        </div>
      );
    };

    const generateInputHandles = (
      schema: BlockIORootSchema,
      nodeType: BlockUIType,
    ) => {
      if (!schema?.properties) return null;
      const keys = Object.entries(schema.properties);
      switch (nodeType) {
        case BlockUIType.NOTE:
          // For NOTE blocks, don't render any input handles
          const [noteKey, noteSchema] = keys[0];
          return (
            <div key={noteKey}>
              <NodeTextBoxInput
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
            nodeType == BlockUIType.AGENT ? `inputs.${key}` : key;

          return keys.map(([propKey, propSchema]) => {
            const isRequired = data.inputSchema.required?.includes(propKey);
            const isAdvanced = propSchema.advanced;
            const isHidden = propSchema.hidden;
            const isConnectable =
              // No input connection handles on INPUT and WEBHOOK blocks
              ![
                BlockUIType.INPUT,
                BlockUIType.WEBHOOK,
                BlockUIType.WEBHOOK_MANUAL,
              ].includes(nodeType) &&
              // No input connection handles for credentials
              propKey !== "credentials" &&
              !propKey.endsWith("_credentials") &&
              // For OUTPUT blocks, only show the 'value' (hides 'name') input connection handle
              !(nodeType == BlockUIType.OUTPUT && propKey == "name");
            const isConnected = isInputHandleConnected(propKey);
            return (
              !isHidden &&
              (isRequired || isAdvancedOpen || isConnected || !isAdvanced) && (
                <div
                  key={propKey}
                  data-id={`input-handle-${propKey}`}
                  className="mb-4"
                >
                  {isConnectable &&
                  !(
                    "oneOf" in propSchema &&
                    propSchema.oneOf &&
                    "discriminator" in propSchema &&
                    propSchema.discriminator
                  ) ? (
                    <NodeHandle
                      keyName={propKey}
                      isConnected={isConnected}
                      isRequired={isRequired}
                      schema={propSchema}
                      side="left"
                    />
                  ) : (
                    propKey !== "credentials" &&
                    !propKey.endsWith("_credentials") && (
                      <div className="flex gap-1">
                        <span className="text-m green mb-0 text-gray-900 dark:text-gray-100">
                          {propSchema.title || beautifyString(propKey)}
                        </span>
                        <InformationTooltip
                          description={propSchema.description}
                        />
                      </div>
                    )
                  )}
                  {isConnected || (
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
    const handleInputChange = useCallback(
      (path: string, value: any) => {
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
      },
      [data.hardcodedValues, id, setHardcodedValues, data.errors, setErrors],
    );

    const isInputHandleConnected = (key: string) => {
      return (
        data.connections &&
        data.connections.some((conn: any) => {
          if (typeof conn === "string") {
            const [_source, target] = conn.split(" -> ");
            return target.includes(key) && target.includes(data.title);
          }
          return conn.target === id && conn.targetHandle === key;
        })
      );
    };

    const isOutputHandleConnected = (key: string) => {
      return (
        data.connections &&
        data.connections.some((conn: any) => {
          if (typeof conn === "string") {
            const [source, _target] = conn.split(" -> ");
            return source.includes(key) && source.includes(data.title);
          }
          return conn.source === id && conn.sourceHandle === key;
        })
      );
    };

    const handleInputClick = useCallback(
      (key: string) => {
        console.debug(`Opening modal for key: ${key}`);
        setActiveKey(key);
        const value = getValue(key, data.hardcodedValues);
        setInputModalValue(
          typeof value === "object" ? JSON.stringify(value, null, 2) : value,
        );
        setIsModalOpen(true);
      },
      [data.hardcodedValues],
    );

    const handleModalSave = useCallback(
      (value: string) => {
        if (activeKey) {
          try {
            const parsedValue = JSON.parse(value);
            // Validate that the parsed value is safe before using it
            if (isObject(parsedValue) || Array.isArray(parsedValue)) {
              handleInputChange(activeKey, parsedValue);
            } else {
              // For primitive values, use the original string
              handleInputChange(activeKey, value);
            }
          } catch {
            // If JSON parsing fails, treat as plain text
            handleInputChange(activeKey, value);
          }
        }
        setIsModalOpen(false);
        setActiveKey(null);
      },
      [activeKey, handleInputChange],
    );

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
          metadata: {
            ...data.metadata,
            customized_name: undefined, // Don't copy the custom name
          },
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
      data.errors && hasNonNullNonObjectValue(data.errors);
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
      "bg-white/[.9] dark:bg-gray-800/[.9]",
      "border border-gray-300 dark:border-gray-600",
      data.uiType === BlockUIType.NOTE ? "w-[300px]" : "w-[500px]",
      data.uiType === BlockUIType.NOTE
        ? "bg-yellow-100 dark:bg-yellow-900"
        : "bg-white dark:bg-gray-800",
      selected ? "shadow-2xl" : "",
    ]
      .filter(Boolean)
      .join(" ");

    const errorClass =
      hasConfigErrors || hasOutputError
        ? "border-red-200 dark:border-red-800 border-2"
        : "";

    const statusClass = (() => {
      if (hasConfigErrors || hasOutputError)
        return "border-red-200 dark:border-red-800 border-4";
      switch (data.status?.toLowerCase()) {
        case "completed":
          return "border-green-200 dark:border-green-800 border-4";
        case "running":
          return "border-yellow-200 dark:border-yellow-800 border-4";
        case "failed":
          return "border-red-200 dark:border-red-800 border-4";
        case "incomplete":
          return "border-purple-200 dark:border-purple-800 border-4";
        case "queued":
          return "border-cyan-200 dark:border-cyan-800 border-4";
        default:
          return "";
      }
    })();

    const statusBackgroundClass = (() => {
      if (hasConfigErrors || hasOutputError)
        return "bg-red-200 dark:bg-red-800";
      switch (data.status?.toLowerCase()) {
        case "completed":
          return "bg-green-200 dark:bg-green-800";
        case "running":
          return "bg-yellow-200 dark:bg-yellow-800";
        case "failed":
          return "bg-red-200 dark:bg-red-800";
        case "incomplete":
          return "bg-purple-200 dark:bg-purple-800";
        case "queued":
          return "bg-cyan-200 dark:bg-cyan-800";
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

    const isCostFilterMatch = (costFilter: any, inputValues: any): boolean => {
      /*
      Filter rules:
      - If costFilter is an object, then check if costFilter is the subset of inputValues
      - Otherwise, check if costFilter is equal to inputValues.
      - Undefined, null, and empty string are considered as equal.
    */
      return typeof costFilter === "object" && typeof inputValues === "object"
        ? Object.entries(costFilter).every(
            ([k, v]) =>
              (!v && !inputValues[k]) || isCostFilterMatch(v, inputValues[k]),
          )
        : costFilter === inputValues;
    };

    const blockCost =
      data.blockCosts &&
      data.blockCosts.find((cost) =>
        isCostFilterMatch(cost.cost_filter, inputValues),
      );

    const LineSeparator = () => (
      <div className="bg-white pt-6 dark:bg-gray-800">
        <Separator.Root className="h-[1px] w-full bg-gray-300 dark:bg-gray-600"></Separator.Root>
      </div>
    );

    const ContextMenuContent = () => (
      <ContextMenu.Content className="z-10 rounded-xl border bg-white p-1 shadow-md dark:bg-gray-800">
        <ContextMenu.Item
          onSelect={copyNode}
          className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <CopyIcon className="mr-2 h-5 w-5 dark:text-gray-100" />
          <span className="dark:text-gray-100">Copy</span>
        </ContextMenu.Item>
        {subGraphID && (
          <ContextMenu.Item
            onSelect={() => window.open(`/build?flowID=${subGraphID}`)}
            className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <ExitIcon className="mr-2 h-5 w-5 dark:text-gray-100" />
            <span className="dark:text-gray-100">Open agent</span>
          </ContextMenu.Item>
        )}
        <ContextMenu.Separator className="my-1 h-px bg-gray-300 dark:bg-gray-600" />
        <ContextMenu.Item
          onSelect={deleteNode}
          className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <TrashIcon className="mr-2 h-5 w-5 text-red-500 dark:text-red-400" />
          <span className="dark:text-red-400">Delete</span>
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
        id={data.block_id}
        z-index={1}
        data-testid={data.block_id}
        data-blockid={data.block_id}
        data-blockname={data.title}
        data-blocktype={data.blockType}
        data-nodetype={data.uiType}
        data-category={data.categories[0]?.category.toLowerCase() || ""}
        data-inputs={JSON.stringify(
          Object.keys(data.inputSchema?.properties || {}),
        )}
        data-outputs={JSON.stringify(
          Object.keys(data.outputSchema?.properties || {}),
        )}
      >
        {/* Header */}
        <div
          className={`flex h-24 border-b border-gray-300 ${data.uiType === BlockUIType.NOTE ? "bg-yellow-100" : "bg-white"} space-x-1 rounded-t-xl`}
        >
          {/* Color Stripe */}
          <div
            className={`-ml-px h-full w-3 rounded-tl-xl ${stripeColor}`}
          ></div>

          <div className="flex w-full flex-col justify-start space-y-2.5 px-4 pt-4">
            <div className="flex flex-row items-center space-x-2 font-semibold">
              <div
                className="group flex items-center gap-1"
                onMouseEnter={() => setIsTitleHovered(true)}
                onMouseLeave={() => setIsTitleHovered(false)}
              >
                {isEditingTitle ? (
                  <Input
                    ref={titleInputRef}
                    value={customTitle}
                    onChange={(e) => setCustomTitle(e.target.value)}
                    onBlur={handleTitleSave}
                    onKeyDown={handleTitleKeyDown}
                    className="h-7 w-auto min-w-[100px] max-w-[200px] px-2 py-1 text-lg font-semibold"
                    placeholder={beautifyString(
                      data.blockType?.replace(/Block$/, "") || data.title,
                    )}
                  />
                ) : (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <h3 className="font-roboto sentry-unmask cursor-default text-lg">
                        <TextRenderer
                          value={displayTitle}
                          truncateLengthLimit={80}
                        />
                      </h3>
                    </TooltipTrigger>
                    {customTitle && (
                      <TooltipContent>
                        <p>
                          Type:{" "}
                          {beautifyString(
                            data.blockType?.replace(/Block$/, "") || data.title,
                          )}
                        </p>
                      </TooltipContent>
                    )}
                  </Tooltip>
                )}
                {isTitleHovered && !isEditingTitle && (
                  <button
                    onClick={handleTitleEdit}
                    className="cursor-pointer rounded p-1 opacity-0 transition-opacity hover:bg-gray-100 group-hover:opacity-100"
                    aria-label="Edit title"
                  >
                    <Pencil1Icon className="h-4 w-4" />
                  </button>
                )}
              </div>
              <span className="text-xs text-gray-500">#{id.split("-")[0]}</span>

              <div className="w-auto grow" />

              <button
                aria-label="Options"
                className="cursor-pointer rounded-full border-none bg-transparent p-1 hover:bg-gray-100"
                onClick={onContextButtonTrigger}
              >
                <DotsVerticalIcon className="h-5 w-5" />
              </button>
            </div>
            <div className="flex items-center space-x-2">
              {blockCost && (
                <div className="mr-3 text-base font-light">
                  <span className="ml-auto flex items-center">
                    <IconCoin />{" "}
                    <span className="mx-1 font-medium">
                      {formatCredits(blockCost.cost_amount)}
                    </span>
                    {" \/ "}
                    {blockCost.cost_type}
                  </span>
                </div>
              )}
              {data.categories.map((category) => (
                <Badge
                  key={category.category}
                  variant="outline"
                  className={`${getPrimaryCategoryColor([category])} h-6 whitespace-nowrap rounded-full border border-gray-300 opacity-50`}
                >
                  {beautifyString(category.category.toLowerCase())}
                </Badge>
              ))}
            </div>
          </div>

          <ContextMenuContent />
        </div>

        {/* Body */}
        <div className="mx-5 my-6 rounded-b-xl">
          {/* Input Handles */}
          {data.uiType !== BlockUIType.NOTE ? (
            <div data-id="input-handles" className="mb-4">
              <div>
                {data.uiType === BlockUIType.AYRSHARE ? (
                  <>
                    {generateAyrshareSSOHandles()}
                    {generateInputHandles(
                      data.inputSchema,
                      BlockUIType.STANDARD,
                    )}
                  </>
                ) : [BlockUIType.WEBHOOK, BlockUIType.WEBHOOK_MANUAL].includes(
                    data.uiType,
                  ) ? (
                  <>
                    <Alert className="mb-3 select-none">
                      <AlertDescription className="flex items-center">
                        <InfoIcon className="mr-2 size-6" />
                        <span>
                          You can set up and manage this trigger in your{" "}
                          <Link
                            href={
                              libraryAgent
                                ? `/library/agents/${libraryAgent.id}`
                                : "/library"
                            }
                            className="underline"
                          >
                            Agent Library
                          </Link>
                          {!data.backend_id && " (after saving the graph)"}.
                        </span>
                      </AlertDescription>
                    </Alert>
                    <div className="pointer-events-none opacity-50">
                      {generateInputHandles(data.inputSchema, data.uiType)}
                    </div>
                  </>
                ) : (
                  data.inputSchema &&
                  generateInputHandles(data.inputSchema, data.uiType)
                )}
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
                />
              </div>
            </>
          )}
          {/* Output Handles */}
          {data.uiType !== BlockUIType.NOTE && (
            <>
              <LineSeparator />
              <div className="flex items-start justify-end rounded-b-xl pt-6">
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
                          "border-red-600 bg-red-600 text-white": [
                            "FAILED",
                            "TERMINATED",
                          ].includes(data.status || ""),
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
  },
  (prevProps, nextProps) => {
    // Only re-render if the 'data' prop has changed
    return prevProps.data === nextProps.data;
  },
);
