import { Calendar } from "@/components/__legacy__/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { format } from "date-fns";
import { CalendarIcon } from "lucide-react";
import { beautifyString, cn } from "@/lib/utils";
import { Node, useNodeId, useNodesData } from "@xyflow/react";
import {
  ConnectionData,
  CustomNodeData,
} from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import { Cross2Icon, Pencil2Icon, PlusIcon } from "@radix-ui/react-icons";
import {
  BlockIOArraySubSchema,
  BlockIOBooleanSubSchema,
  BlockIOCredentialsSubSchema,
  BlockIODiscriminatedOneOfSubSchema,
  BlockIOKVSubSchema,
  BlockIONumberSubSchema,
  BlockIOObjectSubSchema,
  BlockIORootSchema,
  BlockIOSimpleTypeSubSchema,
  BlockIOStringSubSchema,
  BlockIOSubSchema,
  BlockIOTableSubSchema,
  DataType,
  determineDataType,
} from "@/lib/autogpt-server-api/types";
import React, {
  FC,
  useCallback,
  useEffect,
  useMemo,
  useState,
  useRef,
} from "react";
import { Button } from "../../../../../components/__legacy__/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../../../components/__legacy__/ui/select";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "../../../../../components/__legacy__/ui/multiselect";
import { LocalValuedInput } from "../../../../../components/__legacy__/ui/input";
import NodeHandle from "./NodeHandle";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { Switch } from "../../../../../components/atoms/Switch/Switch";
import { NodeTableInput } from "../../../../../components/node-table-input";

type NodeObjectInputTreeProps = {
  nodeId: string;
  selfKey?: string;
  schema: BlockIORootSchema | BlockIOObjectSubSchema;
  object?: { [key: string]: any };
  connections: ConnectionData;
  handleInputClick: (key: string) => void;
  handleInputChange: (key: string, value: any) => void;
  errors: { [key: string]: string | undefined };
  className?: string;
  displayName?: string;
};

const NodeObjectInputTree: FC<NodeObjectInputTreeProps> = ({
  nodeId,
  selfKey = "",
  schema,
  object,
  connections,
  handleInputClick,
  handleInputChange,
  errors,
  className,
}) => {
  object ||= ("default" in schema ? schema.default : null) ?? {};
  return schema.properties ? (
    <div className={cn(className, "w-full flex-col")}>
      {Object.entries(schema.properties).map(([propKey, propSchema]) => {
        const childKey = selfKey ? `${selfKey}.${propKey}` : propKey;

        return (
          <div
            key={propKey}
            className="flex w-full flex-row justify-between space-y-2"
          >
            <span className="mr-2 mt-3 dark:text-gray-300">
              {propSchema.title || beautifyString(propKey)}
            </span>
            <NodeGenericInputField
              nodeId={nodeId}
              key={propKey}
              propKey={childKey}
              propSchema={propSchema}
              currentValue={object ? object[propKey] : undefined}
              errors={errors}
              connections={connections}
              handleInputChange={handleInputChange}
              handleInputClick={handleInputClick}
              displayName={propSchema.title || beautifyString(propKey)}
              parentContext={object}
            />
          </div>
        );
      })}
    </div>
  ) : null;
};

export default NodeObjectInputTree;

const NodeDateTimeInput: FC<{
  selfKey: string;
  schema: BlockIOStringSubSchema;
  value?: string;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
  displayName: string;
  hideDate?: boolean;
  hideTime?: boolean;
}> = ({
  selfKey,
  value = "",
  error,
  handleInputChange,
  className,
  hideDate = false,
  hideTime = false,
}) => {
  const dateInput = value && !hideDate ? new Date(value) : new Date();
  const timeInput = value && !hideTime ? format(dateInput, "HH:mm") : "00:00";

  const handleDateSelect = (newDate: Date | undefined) => {
    if (!newDate) return;

    if (hideTime) {
      // Only pass YYYY-MM-DD if time is hidden
      handleInputChange(selfKey, format(newDate, "yyyy-MM-dd"));
    } else {
      // Otherwise pass full date/time, but still incorporate time
      const [hours, minutes] = timeInput.split(":").map(Number);
      newDate.setHours(hours, minutes);
      handleInputChange(selfKey, newDate.toISOString());
    }
  };

  const handleTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = e.target.value;

    if (hideDate) {
      // Only pass HH:mm if date is hidden
      handleInputChange(selfKey, newTime);
    } else {
      // Otherwise pass full date/time
      const [hours, minutes] = newTime.split(":").map(Number);
      dateInput.setHours(hours, minutes);
      handleInputChange(selfKey, dateInput.toISOString());
    }
  };

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {hideDate || (
        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              className={cn(
                "w-full justify-start text-left font-normal",
                !value && "text-muted-foreground",
              )}
            >
              <CalendarIcon className="mr-2 h-4 w-4" />
              {value && dateInput ? (
                format(dateInput, "PPP")
              ) : (
                <span>Pick a date</span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="start">
            <Calendar
              mode="single"
              selected={dateInput}
              onSelect={handleDateSelect}
              autoFocus
            />
          </PopoverContent>
        </Popover>
      )}
      {hideTime || (
        <LocalValuedInput
          type="time"
          value={timeInput}
          onChange={handleTimeChange}
          className="w-full"
        />
      )}
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const NodeFileInput: FC<{
  selfKey: string;
  schema: BlockIOStringSubSchema;
  value?: string;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
  displayName: string;
}> = ({
  selfKey,
  value = "",
  error,
  handleInputChange,
  className,
  displayName,
}) => {
  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        const base64String = e.target?.result as string;
        handleInputChange(selfKey, base64String);
      };
      reader.readAsDataURL(file);
    },
    [selfKey, handleInputChange],
  );

  const getFileLabel = useCallback((value: string) => {
    if (value.startsWith("data:")) {
      const matches = value.match(/^data:([^;]+);/);
      if (matches?.[1]) {
        const mimeParts = matches[1].split("/");
        if (mimeParts.length > 1) {
          return `${mimeParts[1].toUpperCase()} file`;
        }
        return `${matches[1]} file`;
      }
    } else {
      const pathParts = value.split(".");
      if (pathParts.length > 1) {
        const ext = pathParts.pop();
        if (ext) return `${ext.toUpperCase()} file`;
      }
    }
    return "File";
  }, []);

  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="nodrag flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => inputRef.current?.click()}
            className="w-full"
          >
            {value ? `Change ${displayName}` : `Upload ${displayName}`}
          </Button>
          {value && (
            <Button
              variant="ghost"
              className="text-red-500 hover:text-red-700"
              onClick={() => {
                if (inputRef.current) inputRef.current.value = "";
                handleInputChange(selfKey, "");
              }}
            >
              <Cross2Icon className="h-4 w-4" />
            </Button>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="*/*"
          onChange={handleFileChange}
          className="hidden"
        />
        {value && (
          <div className="break-all rounded-md border border-gray-300 p-2 dark:border-gray-600">
            <small>{getFileLabel(value)}</small>
          </div>
        )}
      </div>
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

export const NodeGenericInputField: FC<{
  nodeId: string;
  propKey: string;
  propSchema: BlockIOSubSchema;
  currentValue?: any;
  errors: NodeObjectInputTreeProps["errors"];
  connections: NodeObjectInputTreeProps["connections"];
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName?: string;
  parentContext?: { [key: string]: any };
}> = ({
  nodeId,
  propKey,
  propSchema,
  currentValue,
  errors,
  connections,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
  parentContext,
}) => {
  className = cn(className);
  displayName ||= propSchema.title || beautifyString(propKey);

  if (
    "oneOf" in propSchema &&
    "discriminator" in propSchema &&
    propSchema.discriminator
  ) {
    return (
      <NodeOneOfDiscriminatorField
        nodeId={nodeId}
        propKey={propKey}
        propSchema={propSchema}
        currentValue={currentValue}
        defaultValue={propSchema.default}
        errors={errors}
        connections={connections}
        handleInputChange={handleInputChange}
        handleInputClick={handleInputClick}
        className={className}
        displayName={displayName}
      />
    );
  }

  const dt = determineDataType(propSchema);
  switch (dt) {
    case DataType.CREDENTIALS:
      return (
        <NodeCredentialsInput
          selfKey={propKey}
          schema={propSchema as BlockIOCredentialsSubSchema}
          value={currentValue}
          errors={errors}
          className={className}
          handleInputChange={handleInputChange}
        />
      );

    case DataType.DATE:
    case DataType.TIME:
    case DataType.DATE_TIME:
      const hideDate = dt === DataType.TIME;
      const hideTime = dt === DataType.DATE;
      return (
        <NodeDateTimeInput
          selfKey={propKey}
          schema={propSchema as BlockIOStringSubSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
          hideDate={hideDate}
          hideTime={hideTime}
        />
      );

    case DataType.FILE:
      return (
        <NodeFileInput
          selfKey={propKey}
          schema={propSchema as BlockIOStringSubSchema}
          value={currentValue}
          error={errors[propKey]}
          handleInputChange={handleInputChange}
          className={className}
          displayName={displayName}
        />
      );

    case DataType.SELECT:
      return (
        <NodeStringInput
          selfKey={propKey}
          schema={propSchema as BlockIOStringSubSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
        />
      );

    case DataType.MULTI_SELECT:
      const schema = propSchema as BlockIOObjectSubSchema;
      return (
        <NodeMultiSelectInput
          selfKey={propKey}
          schema={schema}
          selection={Object.entries(currentValue || {})
            .filter(([_, v]) => v)
            .map(([k, _]) => k)}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={(key, selection) => {
            // If you want to build an object of booleans from `selection`
            // (like your old code), do it here. Otherwise adapt to your actual UI.
            // Example:
            const subSchema =
              schema.properties || (schema as any).anyOf[0].properties;
            const allKeys = subSchema ? Object.keys(subSchema) : [];
            handleInputChange(
              key,
              Object.fromEntries(
                allKeys.map((opt) => [opt, selection.includes(opt)]),
              ),
            );
          }}
        />
      );

    case DataType.BOOLEAN:
      return (
        <NodeBooleanInput
          selfKey={propKey}
          schema={propSchema as BlockIOBooleanSubSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
        />
      );

    case DataType.NUMBER:
      return (
        <NodeNumberInput
          selfKey={propKey}
          schema={propSchema as BlockIONumberSubSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
        />
      );

    case DataType.TABLE:
      const tableSchema = propSchema as BlockIOTableSubSchema;
      // Extract headers from the schema's items properties
      const headers = tableSchema.items?.properties
        ? Object.keys(tableSchema.items.properties)
        : ["Column 1", "Column 2", "Column 3"];
      return (
        <NodeTableInput
          nodeId={nodeId}
          selfKey={propKey}
          schema={tableSchema}
          headers={headers}
          rows={currentValue}
          errors={errors}
          connections={connections}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
          className={className}
          displayName={displayName}
        />
      );

    case DataType.ARRAY:
      return (
        <NodeArrayInput
          nodeId={nodeId}
          selfKey={propKey}
          schema={propSchema as BlockIOArraySubSchema}
          entries={currentValue}
          errors={errors}
          className={className}
          displayName={displayName}
          connections={connections}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
          parentContext={parentContext}
        />
      );

    case DataType.KEY_VALUE:
      return (
        <NodeKeyValueInput
          nodeId={nodeId}
          selfKey={propKey}
          schema={propSchema as BlockIOKVSubSchema}
          entries={currentValue}
          errors={errors}
          connections={connections}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
          className={className}
          displayName={displayName}
        />
      );

    case DataType.OBJECT:
      return (
        <NodeObjectInputTree
          nodeId={nodeId}
          selfKey={propKey}
          schema={propSchema as any}
          object={currentValue}
          errors={errors}
          className={cn("border-l border-gray-500 pl-2", className)} // visual indent
          displayName={displayName}
          connections={connections}
          handleInputClick={handleInputClick}
          handleInputChange={handleInputChange}
        />
      );

    case DataType.LONG_TEXT:
    case DataType.SHORT_TEXT:
    default:
      return (
        <NodeStringInput
          selfKey={propKey}
          schema={propSchema as BlockIOStringSubSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
        />
      );
  }
};

const NodeOneOfDiscriminatorField: FC<{
  nodeId: string;
  propKey: string;
  propSchema: BlockIODiscriminatedOneOfSubSchema;
  currentValue?: any;
  defaultValue?: any;
  errors: { [key: string]: string | undefined };
  connections: ConnectionData;
  handleInputChange: (key: string, value: any) => void;
  handleInputClick: (key: string) => void;
  className?: string;
  displayName?: string;
}> = ({
  nodeId,
  propKey,
  propSchema,
  currentValue,
  defaultValue,
  errors,
  connections,
  handleInputChange,
  handleInputClick,
  className,
}) => {
  const discriminator = propSchema.discriminator;
  const discriminatorProperty = discriminator.propertyName;

  const variantOptions = useMemo(() => {
    const oneOfVariants = propSchema.oneOf || [];

    return oneOfVariants
      .map((variant) => {
        const variantDiscValue = variant.properties?.[discriminatorProperty]
          ?.const as string; // NOTE: can discriminators only be strings?

        return {
          value: variantDiscValue,
          schema: variant,
        };
      })
      .filter((v) => v.value != null);
  }, [discriminatorProperty, propSchema.oneOf]);

  const initialVariant = defaultValue
    ? variantOptions.find(
        (opt) => defaultValue[discriminatorProperty] === opt.value,
      )
    : currentValue
      ? variantOptions.find(
          (opt) => currentValue[discriminatorProperty] === opt.value,
        )
      : null;

  const [chosenType, setChosenType] = useState<string>(
    initialVariant?.value || "",
  );

  useEffect(() => {
    if (initialVariant && !currentValue) {
      handleInputChange(
        propKey,
        defaultValue || {
          [discriminatorProperty]: initialVariant.value,
        },
      );
    }
  }, []);

  const handleVariantChange = (newType: string) => {
    setChosenType(newType);
    const chosenVariant = variantOptions.find((opt) => opt.value === newType);
    if (chosenVariant) {
      const initialValue = {
        [discriminatorProperty]: newType,
      };
      handleInputChange(propKey, initialValue);
    }
  };

  const chosenVariantSchema = variantOptions.find(
    (opt) => opt.value === chosenType,
  )?.schema;

  function getEntryKey(key: string): string {
    // use someKey for handle purpose (not childKey)
    return `${propKey}_#_${key}`;
  }

  function isConnected(key: string): boolean {
    return connections.some(
      (c) => c.targetHandle === getEntryKey(key) && c.target === nodeId,
    );
  }

  return (
    <div
      className={cn(
        "flex min-w-[400px] max-w-[95%] flex-col space-y-4",
        className,
      )}
    >
      <Select value={chosenType || ""} onValueChange={handleVariantChange}>
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select a type..." />
        </SelectTrigger>
        <SelectContent>
          {variantOptions.map((opt: any) => (
            <SelectItem key={opt.value} value={opt.value}>
              {beautifyString(opt.value)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {chosenVariantSchema && chosenVariantSchema.properties && (
        <div className={cn(className, "w-full flex-col")}>
          {Object.entries(chosenVariantSchema.properties).map(
            ([someKey, childSchema]) => {
              if (someKey === "discriminator") {
                return null;
              }
              const childKey = propKey ? `${propKey}.${someKey}` : someKey; // for history redo/undo purpose
              return (
                <div
                  key={childKey}
                  className="mb-4 flex w-full flex-col justify-between space-y-2"
                >
                  <NodeHandle
                    keyName={getEntryKey(someKey)}
                    schema={childSchema}
                    isConnected={isConnected(getEntryKey(someKey))}
                    isRequired={false}
                    side="left"
                  />

                  {!isConnected(someKey) && (
                    <NodeGenericInputField
                      nodeId={nodeId}
                      key={propKey}
                      propKey={childKey}
                      propSchema={childSchema}
                      currentValue={
                        currentValue
                          ? currentValue[someKey]
                          : defaultValue?.[someKey]
                      }
                      errors={errors}
                      connections={connections}
                      handleInputChange={handleInputChange}
                      handleInputClick={handleInputClick}
                      displayName={beautifyString(someKey)}
                    />
                  )}
                </div>
              );
            },
          )}
        </div>
      )}
    </div>
  );
};

const NodeCredentialsInput: FC<{
  selfKey: string;
  schema: BlockIOCredentialsSubSchema;
  value: any;
  errors: { [key: string]: string | undefined };
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
}> = ({ selfKey, schema, value, errors, handleInputChange, className }) => {
  const nodeInputValues = useNodesData<Node<CustomNodeData>>(useNodeId()!)?.data
    .hardcodedValues;
  return (
    <div className={cn("flex flex-col", className)}>
      <CredentialsInput
        schema={schema}
        siblingInputs={nodeInputValues}
        onSelectCredentials={(credsMeta) =>
          handleInputChange(selfKey, credsMeta)
        }
        selectedCredentials={value}
      />
      {errors[selfKey] && (
        <span className="error-message">{errors[selfKey]}</span>
      )}
    </div>
  );
};

const NodeKeyValueInput: FC<{
  nodeId: string;
  selfKey: string;
  schema: BlockIOKVSubSchema;
  entries?: { [key: string]: string } | { [key: string]: number };
  errors: { [key: string]: string | undefined };
  connections: NodeObjectInputTreeProps["connections"];
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName?: string;
}> = ({
  nodeId,
  selfKey,
  entries,
  schema,
  connections,
  handleInputChange,
  handleInputClick,
  errors,
  className,
  displayName,
}) => {
  const getPairValues = useCallback(() => {
    // Map will preserve the order of entries.
    let inputEntries = entries ?? schema.default;
    if (!inputEntries || typeof inputEntries !== "object") inputEntries = {};

    const defaultEntries = new Map(Object.entries(inputEntries));
    const prefix = `${selfKey}_#_`;
    connections
      .filter((c) => c.targetHandle.startsWith(prefix) && c.target === nodeId)
      .map((c) => c.targetHandle.slice(prefix.length))
      .forEach((k) => !defaultEntries.has(k) && defaultEntries.set(k, ""));

    return Array.from(defaultEntries, ([key, value]) => ({ key, value }));
  }, [entries, schema.default, connections, nodeId, selfKey]);

  const [keyValuePairs, setKeyValuePairs] = useState<
    { key: string; value: any }[]
  >([]);

  useEffect(
    () => setKeyValuePairs(getPairValues()),
    [connections, entries, schema.default, getPairValues],
  );

  function updateKeyValuePairs(newPairs: typeof keyValuePairs) {
    setKeyValuePairs(newPairs);

    handleInputChange(
      selfKey,
      newPairs.reduce((obj, { key, value }) => ({ ...obj, [key]: value }), {}),
    );
  }

  function getEntryKey(key: string): string {
    return `${selfKey}_#_${key}`;
  }
  function isConnected(key: string): boolean {
    return connections.some(
      (c) => c.targetHandle === getEntryKey(key) && c.target === nodeId,
    );
  }

  const propSchema =
    schema.additionalProperties && schema.additionalProperties.type
      ? schema.additionalProperties
      : ({ type: "string" } as BlockIOSimpleTypeSubSchema);

  return (
    <div
      className={cn(className, keyValuePairs.length > 0 ? "flex flex-col" : "")}
    >
      <div>
        {keyValuePairs.map(({ key, value }, index) => (
          // The `index` is used as a DOM key instead of the actual `key`
          // because the `key` can change with each input, causing the input to lose focus.
          <div key={index}>
            <NodeHandle
              title={`#${key}`}
              className="text-sm text-gray-500"
              keyName={getEntryKey(key)}
              schema={{ type: "string" }}
              isConnected={isConnected(key)}
              isRequired={false}
              side="left"
            />
            {!isConnected(key) && (
              <div className="nodrag mb-2 flex items-center space-x-2">
                <LocalValuedInput
                  type="text"
                  placeholder="Key"
                  value={key ?? ""}
                  onChange={(e) =>
                    updateKeyValuePairs(
                      keyValuePairs.toSpliced(index, 1, {
                        key: e.target.value,
                        value: value,
                      }),
                    )
                  }
                />
                <NodeGenericInputField
                  className="w-full"
                  nodeId={nodeId}
                  propKey={`${selfKey}_#_${key}`}
                  propSchema={propSchema}
                  currentValue={value}
                  errors={errors}
                  connections={connections}
                  displayName={displayName || beautifyString(key)}
                  handleInputChange={(_, newValue) =>
                    updateKeyValuePairs(
                      keyValuePairs.toSpliced(index, 1, {
                        key: key,
                        value: newValue,
                      }),
                    )
                  }
                  handleInputClick={handleInputClick}
                />
                <Button
                  variant="ghost"
                  className="px-2"
                  onClick={() =>
                    updateKeyValuePairs(keyValuePairs.toSpliced(index, 1))
                  }
                >
                  <Cross2Icon />
                </Button>
              </div>
            )}
            {errors[`${selfKey}.${key}`] && (
              <span className="error-message">
                {errors[`${selfKey}.${key}`]}
              </span>
            )}
          </div>
        ))}
        <Button
          className="bg-gray-200 font-normal text-black hover:text-white dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
          disabled={
            keyValuePairs.length > 0 &&
            !keyValuePairs[keyValuePairs.length - 1].key
          }
          onClick={() =>
            updateKeyValuePairs(keyValuePairs.concat({ key: "", value: "" }))
          }
        >
          <PlusIcon className="mr-2" /> Add Property
        </Button>
      </div>
      {errors[selfKey] && (
        <span className="error-message">{errors[selfKey]}</span>
      )}
    </div>
  );
};

const NodeArrayInput: FC<{
  nodeId: string;
  selfKey: string;
  schema: BlockIOArraySubSchema;
  entries?: string[];
  errors: { [key: string]: string | undefined };
  connections: NodeObjectInputTreeProps["connections"];
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName?: string;
  parentContext?: { [key: string]: any };
}> = ({
  nodeId,
  selfKey,
  schema,
  entries,
  errors,
  connections,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
  parentContext: _parentContext,
}) => {
  entries ??= schema.default;
  if (!entries || !Array.isArray(entries)) entries = [];

  const prefix = `${selfKey}_$_`;
  connections
    .filter((c) => c.targetHandle.startsWith(prefix) && c.target === nodeId)
    .map((c) => parseInt(c.targetHandle.slice(prefix.length)))
    .filter((c) => !isNaN(c))
    .forEach(
      (c) =>
        entries.length <= c &&
        entries.push(...Array(c - entries.length + 1).fill("")),
    );

  const isItemObject = "items" in schema && "properties" in schema.items!;
  const error =
    typeof errors[selfKey] === "string" ? errors[selfKey] : undefined;
  return (
    <div className={cn(className, "flex flex-col")}>
      {entries.map((entry: any, index: number) => {
        const entryKey = `${selfKey}_$_${index}`;
        const isConnected =
          connections &&
          connections.some(
            (c) => c.targetHandle === entryKey && c.target === nodeId,
          );
        return (
          <div key={entryKey}>
            <NodeHandle
              title={`#${index + 1}`}
              className="text-sm text-gray-500"
              keyName={entryKey}
              schema={schema.items!}
              isConnected={isConnected}
              isRequired={false}
              side="left"
            />
            <div className="mb-2 flex space-x-2">
              {!isConnected &&
                (schema.items ? (
                  <NodeGenericInputField
                    className="w-full"
                    nodeId={nodeId}
                    propKey={entryKey}
                    propSchema={schema.items}
                    currentValue={entry}
                    errors={errors}
                    connections={connections}
                    displayName={displayName || beautifyString(selfKey)}
                    handleInputChange={handleInputChange}
                    handleInputClick={handleInputClick}
                  />
                ) : (
                  <NodeFallbackInput
                    selfKey={entryKey}
                    schema={schema.items}
                    value={entry}
                    error={errors[entryKey]}
                    displayName={displayName || beautifyString(selfKey)}
                    handleInputChange={handleInputChange}
                    handleInputClick={handleInputClick}
                  />
                ))}
              {!isConnected && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() =>
                    handleInputChange(selfKey, entries.toSpliced(index, 1))
                  }
                >
                  <Cross2Icon />
                </Button>
              )}
            </div>
            {errors[entryKey] && typeof errors[entryKey] === "string" && (
              <span className="error-message">{errors[entryKey]}</span>
            )}
          </div>
        );
      })}
      <Button
        className="w-[183p] bg-gray-200 font-normal text-black hover:text-white dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
        onClick={() =>
          handleInputChange(selfKey, [...entries, isItemObject ? {} : ""])
        }
      >
        <PlusIcon className="mr-2" /> Add Item
      </Button>
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const NodeMultiSelectInput: FC<{
  selfKey: string;
  schema: BlockIOObjectSubSchema; // TODO: Support BlockIOArraySubSchema
  selection?: string[];
  error?: string;
  className?: string;
  displayName?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
}> = ({
  selfKey,
  schema,
  selection = [],
  error,
  className,
  displayName,
  handleInputChange,
}) => {
  const optionSchema =
    schema.properties ||
    ((schema as any).anyOf?.length > 0
      ? (schema as any).anyOf[0].properties
      : {});
  const options = Object.keys(optionSchema);

  return (
    <div className={cn("flex flex-col", className)}>
      <MultiSelector
        className="nodrag"
        values={selection}
        onValuesChange={(v) => handleInputChange(selfKey, v)}
      >
        <MultiSelectorTrigger>
          <MultiSelectorInput
            placeholder={
              schema.placeholder ?? `Select ${displayName || schema.title}...`
            }
          />
        </MultiSelectorTrigger>
        <MultiSelectorContent className="nowheel">
          <MultiSelectorList>
            {options
              .map((key) => ({ ...optionSchema[key], key }))
              .map(({ key, title, description }) => (
                <MultiSelectorItem key={key} value={key} title={description}>
                  {title ?? key}
                </MultiSelectorItem>
              ))}
          </MultiSelectorList>
        </MultiSelectorContent>
      </MultiSelector>
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const NodeStringInput: FC<{
  selfKey: string;
  schema: BlockIOStringSubSchema;
  value?: string;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName: string;
}> = ({
  selfKey,
  schema,
  value = "",
  error,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
}) => {
  value ||= schema.default || "";
  return (
    <div className={className}>
      {schema.enum && schema.enum.length > 0 ? (
        <Select
          defaultValue={value}
          onValueChange={(newValue) => handleInputChange(selfKey, newValue)}
        >
          <SelectTrigger>
            <SelectValue placeholder={schema.placeholder || displayName} />
          </SelectTrigger>
          <SelectContent className="nodrag">
            {schema.enum
              .filter((option) => option)
              .map((option, index) => (
                <SelectItem key={index} value={option}>
                  {beautifyString(option)}
                </SelectItem>
              ))}
          </SelectContent>
        </Select>
      ) : (
        <div
          className="nodrag relative"
          onClick={schema.secret ? () => handleInputClick(selfKey) : undefined}
        >
          <LocalValuedInput
            type="text"
            id={selfKey}
            value={schema.secret && value ? "*".repeat(value.length) : value}
            onChange={(e) => handleInputChange(selfKey, e.target.value)}
            readOnly={schema.secret}
            placeholder={
              schema?.placeholder || `Enter ${beautifyString(displayName)}`
            }
            className="pr-8 read-only:cursor-pointer read-only:text-gray-500"
          />
          <Button
            variant="ghost"
            size="icon"
            className="absolute inset-1 left-auto h-7 w-7 rounded-[0.25rem]"
            onClick={() => handleInputClick(selfKey)}
            title="Open a larger textbox input"
          >
            <Pencil2Icon className="m-0 p-0" />
          </Button>
        </div>
      )}
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

export const NodeTextBoxInput: FC<{
  selfKey: string;
  schema: BlockIOStringSubSchema;
  value?: string;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName: string;
}> = ({
  selfKey,
  schema,
  value = "",
  error,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
}) => {
  value ||= schema.default || "";

  const [localValue, setLocalValue] = useState(value || schema.default || "");

  useEffect(() => {
    setLocalValue(value || schema.default || "");
  }, [value, schema.default]);

  return (
    <div className={className}>
      <div
        className="nodrag relative m-0 h-[200px] w-full bg-yellow-100 p-4 dark:bg-yellow-900"
        onClick={schema.secret ? () => handleInputClick(selfKey) : undefined}
      >
        <textarea
          id={selfKey}
          value={schema.secret && localValue ? "********" : localValue}
          onChange={(e) => setLocalValue(e.target.value)}
          onBlur={() => handleInputChange(selfKey, localValue)}
          readOnly={schema.secret}
          placeholder={
            schema?.placeholder || `Enter ${beautifyString(displayName)}`
          }
          className="h-full w-full resize-none overflow-hidden border-none bg-transparent text-lg text-black outline-none dark:text-white"
          style={{
            fontSize: "min(1em, 16px)",
            lineHeight: "1.2",
          }}
        />
      </div>
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const NodeNumberInput: FC<{
  selfKey: string;
  schema: BlockIONumberSubSchema;
  value?: number;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
  displayName?: string;
}> = ({
  selfKey,
  schema,
  value,
  error,
  handleInputChange,
  className,
  displayName,
}) => {
  value ||= schema.default;
  displayName ||= schema.title || beautifyString(selfKey);
  return (
    <div className={className}>
      <div className="nodrag flex items-center justify-between space-x-3">
        <LocalValuedInput
          type="number"
          id={selfKey}
          value={value}
          onChange={(e) =>
            handleInputChange(selfKey, parseFloat(e.target.value))
          }
          placeholder={
            schema.placeholder || `Enter ${beautifyString(displayName)}`
          }
          className="dark:text-white"
        />
      </div>
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const NodeBooleanInput: FC<{
  selfKey: string;
  schema: BlockIOBooleanSubSchema;
  value?: boolean;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
  displayName: string;
}> = ({ selfKey, schema, value, error, handleInputChange, className }) => {
  if (value == null) {
    value = schema.default ?? false;
  }
  return (
    <div className={className}>
      <Switch
        defaultChecked={value}
        onCheckedChange={(v) => handleInputChange(selfKey, v)}
      />
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const NodeFallbackInput: FC<{
  selfKey: string;
  schema?: BlockIOSubSchema;
  value: any;
  error?: string;
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName: string;
}> = ({
  selfKey,
  schema,
  value,
  error,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
}) => {
  value ||= (schema as BlockIOStringSubSchema)?.default;
  return (
    <NodeStringInput
      selfKey={selfKey}
      schema={{ type: "string", ...schema } as BlockIOStringSubSchema}
      value={value}
      error={error}
      handleInputChange={handleInputChange}
      handleInputClick={handleInputClick}
      className={className}
      displayName={displayName}
    />
  );
};
