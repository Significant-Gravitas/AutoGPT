import { RJSFSchema } from "@rjsf/utils";
import { getTypeColor } from "@/lib/utils";

export enum InputType {
  SINGLE_LINE_TEXT = "single-line-text",
  TEXT_AREA = "text-area",
  PASSWORD = "password",
  FILE = "file",
  DATE = "date",
  TIME = "time",
  DATE_TIME = "datetime",
  NUMBER = "number",
  INTEGER = "integer",
  SWITCH = "switch",
  ARRAY_EDITOR = "array-editor",
  SELECT = "select",
  MULTI_SELECT = "multi-select",
  OBJECT_EDITOR = "object-editor",
  ENUM = "enum",
}

// This helper function maps a JSONSchema type to an InputType [help us to determine the type of the input]
export function mapJsonSchemaTypeToInputType(
  schema: RJSFSchema,
): InputType | undefined {
  if (schema.type === "string") {
    if (schema.secret) return InputType.PASSWORD;
    if (schema.format === "date") return InputType.DATE;
    if (schema.format === "time") return InputType.TIME;
    if (schema.format === "date-time") return InputType.DATE_TIME;
    if (schema.format === "long-text") return InputType.TEXT_AREA;
    if (schema.format === "short-text") return InputType.SINGLE_LINE_TEXT;
    if (schema.format === "file") return InputType.FILE;
    return InputType.SINGLE_LINE_TEXT;
  }

  if (schema.type === "number") return InputType.NUMBER;
  if (schema.type === "integer") return InputType.INTEGER;
  if (schema.type === "boolean") return InputType.SWITCH;

  if (schema.type === "array") {
    if (
      schema.items &&
      typeof schema.items === "object" &&
      !Array.isArray(schema.items) &&
      schema.items.enum
    ) {
      return InputType.MULTI_SELECT;
    }
    return InputType.ARRAY_EDITOR;
  }

  if (schema.type === "object") {
    return InputType.OBJECT_EDITOR;
  }

  if (schema.enum) {
    return InputType.SELECT;
  }

  if (schema.type === "null") return;

  if (schema.anyOf || schema.oneOf) {
    return undefined;
  }

  return InputType.SINGLE_LINE_TEXT;
}

// Helper to extract options from schema
export function extractOptions(
  schema: any,
): { value: string; label: string }[] {
  if (schema.enum) {
    return schema.enum.map((value: any) => ({
      value: String(value),
      label: String(value),
    }));
  }

  if (schema.type === "array" && schema.items?.enum) {
    return schema.items.enum.map((value: any) => ({
      value: String(value),
      label: String(value),
    }));
  }

  return [];
}

// get display type and color for schema types [need for type display next to field name]
export const getTypeDisplayInfo = (schema: any) => {
  if (
    schema?.type === "array" &&
    "format" in schema &&
    schema.format === "table"
  ) {
    return {
      displayType: "table",
      colorClass: "text-[#6366f1]",
      hexColor: getTypeColor("array"),
    };
  }

  if (schema?.type === "string" && schema?.format) {
    const formatMap: Record<
      string,
      { displayType: string; colorClass: string; hexColor: string }
    > = {
      file: {
        displayType: "file",
        colorClass: "text-green-500",
        hexColor: getTypeColor("string"),
      },
      date: {
        displayType: "date",
        colorClass: "text-blue-500",
        hexColor: getTypeColor("number"),
      },
      time: {
        displayType: "time",
        colorClass: "text-blue-500",
        hexColor: getTypeColor("number"),
      },
      "date-time": {
        displayType: "datetime",
        colorClass: "text-blue-500",
        hexColor: getTypeColor("number"),
      },
      "long-text": {
        displayType: "text",
        colorClass: "text-green-500",
        hexColor: getTypeColor("string"),
      },
      "short-text": {
        displayType: "text",
        colorClass: "text-green-500",
        hexColor: getTypeColor("string"),
      },
    };

    const formatInfo = formatMap[schema.format];
    if (formatInfo) {
      return formatInfo;
    }
  }

  const typeMap: Record<string, string> = {
    string: "text",
    number: "number",
    integer: "integer",
    boolean: "true/false",
    object: "object",
    array: "list",
    null: "null",
  };

  const displayType = typeMap[schema?.type] || schema?.type || "any";

  const colorMap: Record<string, string> = {
    string: "text-green-500",
    number: "text-blue-500",
    integer: "text-blue-500",
    boolean: "text-yellow-500",
    object: "text-purple-500",
    // Same hue as getTypeColor("array"); kept apart from object's purple.
    array: "text-[#6366f1]",
    null: "text-muted-foreground",
    any: "text-muted-foreground",
  };

  const colorClass = colorMap[schema?.type] || "text-muted-foreground";
  const hexColor = getTypeColor(schema?.type);

  return {
    displayType,
    colorClass,
    hexColor,
  };
};

export function getEdgeColorFromOutputType(
  outputSchema: RJSFSchema | undefined,
  sourceHandle: string,
): { colorClass: string; hexColor: string } {
  const defaultColor = {
    colorClass: "stroke-zinc-500/50",
    hexColor: getTypeColor(undefined),
  };

  if (!outputSchema?.properties) return defaultColor;

  const properties = outputSchema.properties as Record<string, unknown>;
  const handleParts = sourceHandle.split("_#_");
  let currentSchema: Record<string, unknown> = properties;

  for (let i = 0; i < handleParts.length; i++) {
    const part = handleParts[i];
    const fieldSchema = currentSchema[part] as Record<string, unknown>;
    if (!fieldSchema) return defaultColor;

    if (i === handleParts.length - 1) {
      const { hexColor, colorClass } = getTypeDisplayInfo(fieldSchema);
      return { colorClass: colorClass.replace("text-", "stroke-"), hexColor };
    }

    if (fieldSchema.properties) {
      currentSchema = fieldSchema.properties as Record<string, unknown>;
    } else {
      return defaultColor;
    }
  }

  return defaultColor;
}
