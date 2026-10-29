import { RJSFSchema } from "@rjsf/utils";

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
      // oxlint-disable-next-line design-system/no-raw-colors -- each data type keeps its own hue; no role is left that would not collide with another type
      colorClass: "!text-indigo-500",
      // TODO(tokens): no exact role: literal #6366F1
      hexColor: "#6366f1",
    };
  }

  if (schema?.type === "string" && schema?.format) {
    const formatMap: Record<
      string,
      { displayType: string; colorClass: string; hexColor: string }
    > = {
      file: {
        displayType: "file",
        colorClass: "!text-success",
        // TODO(tokens): no exact role: literal #22C55E
        hexColor: "#22c55e",
      },
      date: {
        displayType: "date",
        colorClass: "!text-info",
        // TODO(tokens): no fitting role for literal #3B82F6; #3B82F6 is info (a status colour), sidebar-ring (a focus ring colour)
        hexColor: "#3b82f6",
      },
      time: {
        displayType: "time",
        colorClass: "!text-info",
        // TODO(tokens): no fitting role for literal #3B82F6; #3B82F6 is info (a status colour), sidebar-ring (a focus ring colour)
        hexColor: "#3b82f6",
      },
      "date-time": {
        displayType: "datetime",
        colorClass: "!text-info",
        // TODO(tokens): no fitting role for literal #3B82F6; #3B82F6 is info (a status colour), sidebar-ring (a focus ring colour)
        hexColor: "#3b82f6",
      },
      "long-text": {
        displayType: "text",
        colorClass: "!text-success",
        // TODO(tokens): no exact role: literal #22C55E
        hexColor: "#22c55e",
      },
      "short-text": {
        displayType: "text",
        colorClass: "!text-success",
        // TODO(tokens): no exact role: literal #22C55E
        hexColor: "#22c55e",
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
    string: "!text-success",
    number: "!text-info",
    integer: "!text-info",
    // oxlint-disable-next-line design-system/no-raw-colors -- each data type keeps its own hue; no role is left that would not collide with another type
    boolean: "!text-yellow-500",
    object: "!text-brand",
    // oxlint-disable-next-line design-system/no-raw-colors -- each data type keeps its own hue; no role is left that would not collide with another type
    array: "!text-indigo-500",
    null: "!text-foreground-tertiary",
    any: "!text-foreground-tertiary",
  };

  const hexColorMap: Record<string, string> = {
    // TODO(tokens): no exact role: literal #22C55E
    string: "#22c55e",
    // TODO(tokens): no fitting role for literal #3B82F6; #3B82F6 is info (a status colour), sidebar-ring (a focus ring colour)
    number: "#3b82f6",
    // TODO(tokens): no fitting role for literal #3B82F6; #3B82F6 is info (a status colour), sidebar-ring (a focus ring colour)
    integer: "#3b82f6",
    // TODO(tokens): no exact role: literal #EAB308
    boolean: "#eab308",
    // TODO(tokens): no exact role: literal #A855F7
    object: "#a855f7",
    // TODO(tokens): no exact role: literal #6366F1
    array: "#6366f1",
    // TODO(tokens): no exact role: literal #6B7280
    null: "#6b7280",
    // TODO(tokens): no exact role: literal #6B7280
    any: "#6b7280",
  };

  const colorClass = colorMap[schema?.type] || "!text-foreground-tertiary";
  // TODO(tokens): no exact role: literal #6B7280
  const hexColor = hexColorMap[schema?.type] || "#6b7280";

  return {
    displayType,
    colorClass,
    hexColor,
  };
};

// Whole class names, so Tailwind finds each stroke colour in the source.
const EDGE_STROKE_CLASSES: Record<string, string> = {
  "!text-success": "stroke-success",
  "!text-info": "stroke-info",
  // oxlint-disable-next-line design-system/no-raw-colors -- each data type keeps its own hue; no role is left that would not collide with another type
  "!text-yellow-500": "stroke-yellow-500",
  "!text-brand": "stroke-brand",
  // oxlint-disable-next-line design-system/no-raw-colors -- each data type keeps its own hue; no role is left that would not collide with another type
  "!text-indigo-500": "stroke-indigo-500",
  "!text-foreground-tertiary": "stroke-foreground-tertiary",
};

export function getEdgeColorFromOutputType(
  outputSchema: RJSFSchema | undefined,
  sourceHandle: string,
): { colorClass: string; hexColor: string } {
  const defaultColor = {
    colorClass: "stroke-edge/50",
    // TODO(tokens): no exact role: literal #6B7280
    hexColor: "#6b7280",
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
      return {
        colorClass: EDGE_STROKE_CLASSES[colorClass] ?? defaultColor.colorClass,
        hexColor,
      };
    }

    if (fieldSchema.properties) {
      currentSchema = fieldSchema.properties as Record<string, unknown>;
    } else {
      return defaultColor;
    }
  }

  return defaultColor;
}
