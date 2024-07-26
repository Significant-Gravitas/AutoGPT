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
import { FC, useState } from "react";
import { Button } from "./ui/button";
import { Switch } from "./ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

type NodeObjectInputTreeProps = {
  selfKey?: string;
  schema: BlockIORootSchema | BlockIOObjectSubSchema;
  object?: { [key: string]: any };
  handleInputClick: (key: string) => void;
  handleInputChange: (key: string, value: any) => void;
  errors: { [key: string]: string | undefined };
  className?: string;
  displayName?: string;
};

const NodeObjectInputTree: FC<NodeObjectInputTreeProps> = ({
  selfKey = "",
  schema,
  object,
  handleInputClick,
  handleInputChange,
  errors,
  className,
  displayName,
}) => {
  object ??= ("default" in schema ? schema.default : null) ?? {};
  return (
    <div className={className}>
      {displayName && <strong>{displayName}:</strong>}
      {Object.entries(schema.properties).map(([propKey, propSchema]) => {
        const childKey = selfKey ? `${selfKey}.${propKey}` : propKey;
        return (
          <NodeGenericInputField
            key={propKey}
            propKey={childKey}
            propSchema={propSchema}
            currentValue={object[propKey]}
            errors={errors}
            handleInputChange={handleInputChange}
            handleInputClick={handleInputClick}
          />
        )
      })}
    </div>
  );
};

export default NodeObjectInputTree;

export const NodeGenericInputField: FC<{
  propKey: string;
  propSchema: BlockIOSubSchema;
  currentValue?: any;
  errors: NodeObjectInputTreeProps["errors"];
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName?: string;
}> = ({
  propKey,
  propSchema,
  currentValue,
  errors,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
}) => {
  displayName ??= propSchema.title || beautifyString(propKey);

  if ("allOf" in propSchema) {
    // If this happens, that is because Pydantic wraps $refs in an allOf if the
    // $ref has sibling schema properties (which isn't technically allowed),
    // so there will only be one item in allOf[].
    // AFAIK this should NEVER happen though, as $refs are resolved server-side.
    propSchema = propSchema.allOf[0];
    console.warn(`Unsupported 'allOf' in schema for '${propKey}'!`, propSchema);
  }

  if ("properties" in propSchema) {
    return (
      <NodeObjectInputTree
        selfKey={propKey}
        schema={propSchema}
        object={currentValue}
        errors={errors}
        className={cn("border-l border-gray-500 pl-2", className)}  // visual indent
        displayName={displayName}
        handleInputClick={handleInputClick}
        handleInputChange={handleInputChange}
      />
    );
  }

  if ("additionalProperties" in propSchema) {
    return (
      <NodeKeyValueInput
        selfKey={propKey}
        schema={propSchema}
        entries={currentValue}
        errors={errors}
        className={className}
        displayName={displayName}
        handleInputChange={handleInputChange}
      />
    );
  }

  if ("anyOf" in propSchema) {  // optional items
    const types = propSchema.anyOf.map((s) => ("type" in s ? s.type : undefined));
    if (types.includes("string") && types.includes("null")) {  // optional string
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
    console.error(`Unsupported 'oneOf' in schema for '${propKey}'!`, propSchema);
    return null;
  }

  if (!("type" in propSchema)) {
    console.warn(`Schema for '${propKey}' does not specify a type:`, propSchema);
    return (
      <NodeFallbackInput
        selfKey={propKey}
        schema={propSchema}
        value={currentValue}
        error={errors[propKey]}
        className={className}
        displayName={displayName}
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
          selfKey={propKey}
          schema={propSchema}
          entries={currentValue}
          errors={errors}
          className={className}
          displayName={displayName}
          handleInputChange={handleInputChange}
          handleInputClick={handleInputClick}
        />
      );
    default:
      console.warn(`Schema for '${propKey}' specifies unknown type:`, propSchema);
      return (
        <NodeFallbackInput
          selfKey={propKey}
          schema={propSchema}
          value={currentValue}
          error={errors[propKey]}
          className={className}
          displayName={displayName}
          handleInputClick={handleInputClick}
        />
      );
  }
}

const NodeKeyValueInput: FC<{
  selfKey: string;
  schema: BlockIOKVSubSchema;
  entries?: { [key: string]: string } | { [key: string]: number };
  errors: { [key: string]: string | undefined };
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  className?: string;
  displayName?: string;
}> = ({ selfKey, entries, schema, handleInputChange, errors, className, displayName }) => {
  const [keyValuePairs, setKeyValuePairs] = useState<{
    key: string;
    value: string | number | null;
  }[]>(
    Object.entries(entries ?? schema.default ?? {})
      .map(([key, value]) => ({ key, value: value }))
  );

  function updateKeyValuePairs(newPairs: typeof keyValuePairs) {
    setKeyValuePairs(newPairs);
    handleInputChange(
      selfKey,
      newPairs.reduce((obj, { key, value }) => ({ ...obj, [key]: value }), {})
    );
  };

  function convertValueType(value: string): string | number | null {
    if (schema.additionalProperties.type == "string") return value;
    if (!value) return null;
    return Number(value)
  }

  return (
    <div className={className}>
      {displayName && <strong>{displayName}:</strong>}
      <div>
        {keyValuePairs.map(({ key, value }, index) => (
          <div key={index}>
            <div className="flex items-center space-x-2 mb-2">
              <Input
                type="text"
                placeholder="Key"
                value={key}
                onChange={(e) =>
                  updateKeyValuePairs(
                    keyValuePairs.toSpliced(index, 1, {
                      key: e.target.value,
                      value: value,
                    })
                  )
                }
              />
              <Input
                type="text"
                placeholder="Value"
                value={value ?? ""}
                onChange={(e) =>
                  updateKeyValuePairs(
                    keyValuePairs.toSpliced(index, 1, {
                      key: key,
                      value: convertValueType(e.target.value),
                    })
                  )
                }
              />
              <Button
                variant="ghost"
                className="px-2"
                onClick={() => updateKeyValuePairs(keyValuePairs.toSpliced(index, 1))}
              >
                <Cross2Icon />
              </Button>
            </div>
            {errors[`${selfKey}.${key}`] &&
              <span className="error-message">{errors[`${selfKey}.${key}`]}</span>
            }
          </div>
        ))}
        <Button
          className="w-full"
          onClick={() =>
            updateKeyValuePairs(keyValuePairs.concat({ key: "", value: "" }))
          }
        >
          <PlusIcon className="mr-2" /> Add Property
        </Button>
      </div>
      {errors[selfKey] && <span className="error-message">{errors[selfKey]}</span>}
    </div>
  );
};

const NodeArrayInput: FC<{
  selfKey: string;
  schema: BlockIOArraySubSchema;
  entries?: string[];
  errors: { [key: string]: string | undefined };
  handleInputChange: NodeObjectInputTreeProps["handleInputChange"];
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName?: string;
}> = ({
  selfKey,
  schema,
  entries,
  errors,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
}) => {
  entries ??= schema.default ?? [];
  return (
    <div className={`input-container ${className ?? ""}`}>
      {displayName && <strong>{displayName}</strong>}
      {entries.map((entry: string, index: number) => {
        const entryKey = `${selfKey}[${index}]`;
        return (
          <div key={entryKey}>
            <div className="flex items-center space-x-2 mb-2">
              {schema.items
                ? <NodeGenericInputField
                  propKey={entryKey}
                  propSchema={schema.items}
                  currentValue={entry}
                  errors={errors}
                  handleInputChange={handleInputChange}
                  handleInputClick={handleInputClick}
                />
                : <NodeFallbackInput
                  selfKey={entryKey}
                  schema={schema.items}
                  value={entry}
                  error={errors[entryKey]}
                  displayName={displayName ?? "something"}
                  handleInputClick={handleInputClick}
                />
              }
              <Button
                variant="ghost" size="icon"
                onClick={() => handleInputChange(selfKey, entries.toSpliced(index, 1))}
              >
                <Cross2Icon />
              </Button>
            </div>
            {errors[entryKey] &&
              <span className="error-message">{errors[entryKey]}</span>
            }
          </div>
        )
      })}
      <Button onClick={() => handleInputChange(selfKey, [...entries, ""])}>
        <PlusIcon className="mr-2" /> Add Item
      </Button>
      {errors[selfKey] && <span className="error-message">{errors[selfKey]}</span>}
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
  value,
  error,
  handleInputChange,
  handleInputClick,
  className,
  displayName,
}) => {
  return (
    <div className={`input-container ${className ?? ""}`}>
      {schema.enum ? (
        <Select
          defaultValue={value}
          onValueChange={newValue => handleInputChange(selfKey, newValue)}
        >
          <SelectTrigger>
            <SelectValue placeholder={displayName || schema.placeholder || schema.title} />
          </SelectTrigger>
          <SelectContent>
            {schema.enum.map((option, index) => (
              <SelectItem key={index} value={option}>
                {beautifyString(option)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <div className="relative">
          <Input
            type="text"
            id={selfKey}
            value={value}
            onChange={e => handleInputChange(selfKey, e.target.value)}
            className="pr-8"
          />
          <Button
            variant="ghost" size="icon"
            className="absolute inset-1 left-auto h-7 w-7 rounded-[0.25rem]"
            onClick={() => handleInputClick(selfKey)}
            title="Open a larger textbox input"
          >
            <Pencil2Icon />
          </Button>
        </div>
      )}
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
}> = ({ selfKey, schema, value, error, handleInputChange, className, displayName }) => {
  value ??= schema.default;
  return (
    <div className={`input-container ${className ?? ""}`}>
      <div className="flex items-center justify-between space-x-3">
        {displayName && <Label htmlFor={selfKey}>{displayName}</Label>}
        <Input
          type="number"
          id={selfKey}
          value={value}
          onChange={(e) => handleInputChange(selfKey, parseFloat(e.target.value))}
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
}> = ({ selfKey, schema, value, error, handleInputChange, className, displayName }) => {
  value ??= schema.default ?? false;
  return (
    <div className={`input-container ${className ?? ""}`}>
      <div className="flex items-center">
        <Switch checked={value} onCheckedChange={v => handleInputChange(selfKey, v)} />
        <span className="ml-3">{displayName}</span>
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
  handleInputClick: NodeObjectInputTreeProps["handleInputClick"];
  className?: string;
  displayName: string;
}> = ({ selfKey, schema, value, error, handleInputClick, className, displayName }) => {
  return (
    <div className={`input-container ${className ?? ""}`}>
      <ClickableInput
        handleInputClick={() => handleInputClick(selfKey)}
        value={value}
        placeholder={
          schema?.placeholder || `Enter ${beautifyString(displayName)} (Complex)`
        }
      />
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};

const ClickableInput: FC<{
  handleInputClick: () => void;
  value?: string;
  secret?: boolean;
  placeholder?: string;
  className?: string;
}> = ({
  handleInputClick,
  value = "",
  placeholder = "",
  secret = false,
  className,
}) => (
  <div className={`clickable-input ${className}`} onClick={handleInputClick}>
    {secret
      ? <i className="text-gray-500">{value ? "********" : placeholder}</i>
      : value || <i className="text-gray-500">{placeholder}</i>
    }
  </div>
);
