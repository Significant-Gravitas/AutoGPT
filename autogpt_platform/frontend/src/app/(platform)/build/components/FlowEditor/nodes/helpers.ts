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
      colorClass: "!text-indigo",
    };
  }

  if (schema?.type === "string" && schema?.format) {
    const formatMap: Record<
      string,
      { displayType: string; colorClass: string }
    > = {
      file: {
        displayType: "file",
        colorClass: "!text-success",
      },
      date: {
        displayType: "date",
        colorClass: "!text-info",
      },
      time: {
        displayType: "time",
        colorClass: "!text-info",
      },
      "date-time": {
        displayType: "datetime",
        colorClass: "!text-info",
      },
      "long-text": {
        displayType: "text",
        colorClass: "!text-success",
      },
      "short-text": {
        displayType: "text",
        colorClass: "!text-success",
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
    boolean: "!text-yellow",
    object: "!text-brand",
    array: "!text-indigo",
    null: "!text-foreground-tertiary",
    any: "!text-foreground-tertiary",
  };

  const colorClass = colorMap[schema?.type] || "!text-foreground-tertiary";

  return {
    displayType,
    colorClass,
  };
};

// Whole class names, so Tailwind finds each stroke colour in the source.
const EDGE_STROKE_CLASSES: Record<string, string> = {
  "!text-success": "stroke-success",
  "!text-info": "stroke-info",
  "!text-yellow": "stroke-yellow",
  "!text-brand": "stroke-brand",
  "!text-indigo": "stroke-indigo",
  "!text-foreground-tertiary": "stroke-foreground-tertiary",
};

export function getEdgeColorFromOutputType(
  outputSchema: RJSFSchema | undefined,
  sourceHandle: string,
): { colorClass: string } {
  const defaultColor = { colorClass: "stroke-edge/50" };

  if (!outputSchema?.properties) return defaultColor;

  const properties = outputSchema.properties as Record<string, unknown>;
  const handleParts = sourceHandle.split("_#_");
  let currentSchema: Record<string, unknown> = properties;

  for (let i = 0; i < handleParts.length; i++) {
    const part = handleParts[i];
    const fieldSchema = currentSchema[part] as Record<string, unknown>;
    if (!fieldSchema) return defaultColor;

    if (i === handleParts.length - 1) {
      const { colorClass } = getTypeDisplayInfo(fieldSchema);
      return {
        colorClass: EDGE_STROKE_CLASSES[colorClass] ?? defaultColor.colorClass,
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
