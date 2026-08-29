import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import type { OutputMetadata } from "@/components/contextual/OutputRenderers";
import { globalRegistry } from "@/components/contextual/OutputRenderers";
import React from "react";

export type NodeDataType = "input" | "output";

export type OutputItem = {
  key: string;
  value: unknown;
  metadata?: OutputMetadata;
  renderer: any;
};

export const normalizeToArray = (value: unknown) => {
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
};

export const getExecutionData = (
  result: NodeExecutionResult,
  dataType: NodeDataType,
  pinName: string,
) => {
  if (dataType === "input") {
    return result.input_data;
  }

  return result.output_data?.[pinName];
};

export const createOutputItems = (dataArray: unknown[]): Array<OutputItem> => {
  const items: Array<OutputItem> = [];

  dataArray.forEach((value, index) => {
    const metadata: OutputMetadata = {};

    if (
      typeof value === "object" &&
      value !== null &&
      !React.isValidElement(value)
    ) {
      const objValue = value as any;
      if (objValue.type) metadata.type = objValue.type;
      if (objValue.mimeType) metadata.mimeType = objValue.mimeType;
      if (objValue.filename) metadata.filename = objValue.filename;
      if (objValue.language) metadata.language = objValue.language;
    }

    const renderer = globalRegistry.getRenderer(value, metadata);
    if (renderer) {
      items.push({
        key: `item-${index}`,
        value,
        metadata,
        renderer,
      });
    } else {
      const textRenderer = globalRegistry
        .getAllRenderers()
        .find((r) => r.name === "TextRenderer");
      if (textRenderer) {
        items.push({
          key: `item-${index}`,
          value:
            typeof value === "string" ? value : JSON.stringify(value, null, 2),
          metadata,
          renderer: textRenderer,
        });
      }
    }
  });

  return items;
};

export const getExecutionEntries = (
  result: NodeExecutionResult,
  dataType: NodeDataType,
) => {
  const data = dataType === "input" ? result.input_data : result.output_data;
  return Object.entries(data || {});
};
