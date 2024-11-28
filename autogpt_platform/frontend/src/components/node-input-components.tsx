import { Cross2Icon, Pencil2Icon, PlusIcon } from "@radix-ui/react-icons";
import { beautifyString, cn } from "@/lib/utils";
import {
  BlockIORootSchema,
  BlockIOSubSchema,
  BlockIOObjectSubSchema,
  BlockIOKVSubSchema,
  BlockIOArraySubSchema,
  BlockIOStringSubSchema,
  BlockIONumberSubSchema,
  BlockIOBooleanSubSchema,
} from "@/lib/autogpt-server-api/types";
import React, { FC, useCallback, useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Switch } from "./ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "./ui/multiselect";
import { LocalValuedInput } from "./ui/input";
import NodeHandle from "./NodeHandle";
import { ConnectionData } from "./CustomNode";
import { CredentialsInput } from "./integrations/credentials-input";

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
  displayName,
}) => {
  object ||= ("default" in schema ? schema.default : null) ?? {};
  return (
    <div className={cn(className, "w-full flex-col")}>
      {Object.entries(schema.properties).map(([propKey, propSchema]) => {
        const childKey = selfKey ? `${selfKey}.${propKey}` : propKey;

        return (
          <div
            key={propKey}
            className="flex w-full flex-row justify-between space-y-2"
          >
            <span className="mr-2 mt-3">
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
            />
          </div>
        );
      })}
    </div>
  );
};

export default NodeObjectInputTree;

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
}) => {
  className = cn(className, "my-2");
  displayName ||= propSchema.title || beautifyString(propKey);

  if ("allOf" in propSchema) {
    // If this happens, that is because Pydantic wraps $refs in an allOf if the
    // $ref has sibling schema properties (which isn't technically allowed),
    // so there will only be one item in allOf[].
    // AFAIK this should NEVER happen though, as $refs are resolved server-side.
    propSchema = propSchema.allOf[0];
    console.warn(`Unsupported 'allOf' in schema for '${propKey}'!`, propSchema);
  }

  if ("credentials_provider" in propSchema) {
    return (
      <NodeCredentialsInput
        selfKey={propKey}
        value={currentValue}
        errors={errors}
        className={className}
        handleInputChange={handleInputChange}
      />
    );
  }

  if ("properties" in propSchema) {
    // Render a multi-select for all-boolean sub-schemas with more than 3 properties
    if (
      Object.values(propSchema.properties).every(
        (subSchema) => "type" in subSchema && subSchema.type == "boolean",
      ) &&
      Object.keys(propSchema.properties).length >= 3
    ) {
      const options = Object.keys(propSchema.properties);
      const selectedKeys = Object.entries(currentValue || {})
        .filter(([_, v]) => v)
        .map(([k, _]) => k);
      return (
        <NodeMultiSelectInput
          selfKey={propKey}
          schema={propSchema}
          selection={selectedKeys}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={(key, selection) => {
            handleInputChange(
              key,
              Object.fromEntries(
                options.map((option) => [option, selection.includes(option)]),
              ),
            );
          }}
        />
      );
    }

    return (
      <NodeObjectInputTree
        nodeId={nodeId}
        selfKey={propKey}
        schema={propSchema}
        object={currentValue}
        errors={errors}
        className={cn("border-l border-gray-500 pl-2", className)} // visual indent
        displayName={displayName}
        connections={connections}
        handleInputClick={handleInputClick}
        handleInputChange={handleInputChange}
      />
    );
  }

  if ("additionalProperties" in propSchema) {
    return (
      <NodeKeyValueInput
        nodeId={nodeId}
        selfKey={propKey}
        schema={propSchema}
        entries={currentValue}
        errors={errors}
        className={className}
        displayName={displayName}
        connections={connections}
        handleInputChange={handleInputChange}
      />
    );
  }

  if ("anyOf" in propSchema) {
    // optional items
    const types = propSchema.anyOf.map((s) =>
      "type" in s ? s.type : undefined,
    );
    if (types.includes("string") && types.includes("null")) {
      // optional string
      return (
        <NodeStringInput
          selfKey={propKey}
          schema={{ ...propSchema, type: "string" } as BlockIOStringSubSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
        />
      );
    }
  }

  if ("oneOf" in propSchema) {
    // At the time of writing, this isn't used in the backend -> no impl. needed
    console.error(
      `Unsupported 'oneOf' in schema for '${propKey}'!`,
      propSchema,
    );
    return null;
  }

  if (!("type" in propSchema)) {
    return (
      <NodeFallbackInput
        selfKey={propKey}
        schema={propSchema}
        value={currentValue}
        error={errors[propKey]}
        className={className}
        displayName={displayName}
        handleInputChange={handleInputChange}
        handleInputClick={handleInputClick}
      />
    );
  }

  switch (propSchema.type) {
    case "string":
      return (
        <NodeStringInput
          selfKey={propKey}
          schema={propSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
        />
      );
    case "boolean":
      return (
        <NodeBooleanInput
          selfKey={propKey}
          schema={propSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
        />
      );
    case "number":
    case "integer":
      return (
        <NodeNumberInput
          selfKey={propKey}
          schema={propSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
        />
      );
    case "array":
      return (
        <NodeArrayInput
          nodeId={nodeId}
          selfKey={propKey}
          schema={propSchema}
          entries={currentValue}
          errors={errors}
          className={className}
          displayName={displayName}
          connections={connections}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
        />
      );
    case "object":
      return (
        <NodeKeyValueInput
          nodeId={nodeId}
          selfKey={propKey}
          schema={propSchema}
          entries={currentValue}
          errors={errors}
          className={className}
          displayName={displayName}
          connections={connections}
          handleInputChange={handleInputChange}
        />
      );
    default:
      console.warn(
        `Schema for '${propKey}' specifies unknown type:`,
        propSchema,
      );
      return (
        <NodeFallbackInput
          selfKey={propKey}
          schema={propSchema}
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

const NodeCredentialsInput: FC<{
  selfKey: string;
  value: any;
  errors: { [key: string]: string | undefined };
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
}> = ({ selfKey, value, errors, handleInputChange, className }) => {
  return (
    <div className={cn("flex flex-col", className)}>
      <CredentialsInput
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

const InputRef = (value: any): ((el: HTMLInputElement | null) => void) => {
  return (el) => el && value != null && (el.value = value);
};

const NodeKeyValueInput: FC<{
  nodeId: string;
  selfKey: string;
  schema: BlockIOKVSubSchema;
  entries?: { [key: string]: string } | { [key: string]: number };
  errors: { [key: string]: string | undefined };
  connections: NodeObjectInputTreeProps["connections"];
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
  displayName?: string;
}> = ({
  nodeId,
  selfKey,
  entries,
  schema,
  connections,
  handleInputChange,
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
    { key: string; value: string | number | null }[]
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

  function convertValueType(value: string): string | number | null {
    if (
      !schema.additionalProperties ||
      schema.additionalProperties.type == "string"
    )
      return value;
    if (!value) return null;
    return Number(value);
  }

  function getEntryKey(key: string): string {
    return `${selfKey}_#_${key}`;
  }
  function isConnected(key: string): boolean {
    return connections.some(
      (c) => c.targetHandle === getEntryKey(key) && c.target === nodeId,
    );
  }

  return (
    <div
      className={cn(className, keyValuePairs.length > 0 ? "flex flex-col" : "")}
    >
      <div>
        {keyValuePairs.map(({ key, value }, index) => (
          /* 
          The `index` is used as a DOM key instead of the actual `key`
          because the `key` can change with each input, causing the input to lose focus.
          */
          <div key={index}>
            <NodeHandle
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
                <LocalValuedInput
                  type="text"
                  placeholder="Value"
                  value={value ?? ""}
                  onChange={(e) =>
                    updateKeyValuePairs(
                      keyValuePairs.toSpliced(index, 1, {
                        key: key,
                        value: convertValueType(e.target.value),
                      }),
                    )
                  }
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
          className="bg-gray-200 font-normal text-black hover:text-white"
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
          <div key={entryKey} className="self-start">
            <div className="mb-2 flex space-x-2">
              <NodeHandle
                keyName={entryKey}
                schema={schema.items!}
                isConnected={isConnected}
                isRequired={false}
                side="left"
              />
              {!isConnected &&
                (schema.items ? (
                  <NodeGenericInputField
                    nodeId={nodeId}
                    propKey={entryKey}
                    propSchema={schema.items}
                    currentValue={entry}
                    errors={errors}
                    connections={connections}
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
        className="w-[183p] bg-gray-200 font-normal text-black hover:text-white"
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
  const options = Object.keys(schema.properties);

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
              .map((key) => ({ ...schema.properties[key], key }))
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
  if (!value) {
    value = schema.default || "";
    // Force update hardcodedData so discriminators can update
    // e.g. credentials update when provider changes
    // this won't happen if the value is only set here to schema.default
    if (schema.default) {
      handleInputChange(selfKey, value);
    }
  }
  return (
    <div className={className}>
      {schema.enum ? (
        <Select
          defaultValue={value}
          onValueChange={(newValue) => handleInputChange(selfKey, newValue)}
        >
          <SelectTrigger>
            <SelectValue placeholder={schema.placeholder || displayName} />
          </SelectTrigger>
          <SelectContent className="nodrag">
            {schema.enum.map((option, index) => (
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
  return (
    <div className={className}>
      <div
        className="nodrag relative m-0 h-[200px] w-full bg-yellow-100 p-4"
        onClick={schema.secret ? () => handleInputClick(selfKey) : undefined}
      >
        <textarea
          id={selfKey}
          value={schema.secret && value ? "********" : value}
          readOnly={schema.secret}
          placeholder={
            schema?.placeholder || `Enter ${beautifyString(displayName)}`
          }
          onChange={(e) => handleInputChange(selfKey, e.target.value)}
          className="h-full w-full resize-none overflow-hidden border-none bg-transparent text-lg text-black outline-none"
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
}> = ({
  selfKey,
  schema,
  value,
  error,
  handleInputChange,
  className,
  displayName,
}) => {
  value ||= schema.default ?? false;
  return (
    <div className={className}>
      <div className="nodrag flex items-center">
        <Switch
          defaultChecked={value}
          onCheckedChange={(v) => handleInputChange(selfKey, v)}
        />
        {displayName && <span className="ml-3">{displayName}</span>}
      </div>
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
