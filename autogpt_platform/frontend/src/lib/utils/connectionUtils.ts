import { Block, BlockIOSubSchema } from "@/lib/autogpt-server-api";

/**
 * Check if two types are compatible for connection
 * @param sourceType The output type from source node
 * @param targetType The input type from target node
 * @returns true if types are compatible
 */
export function areTypesCompatible(
  sourceType: string | undefined,
  targetType: string | undefined,
): boolean {
  // Any type can connect to anything
  if (sourceType === "any" || targetType === "any") {
    return true;
  }

  // If either is undefined, allow connection (legacy behavior)
  if (!sourceType || !targetType) {
    return true;
  }

  // Direct type match
  if (sourceType === targetType) {
    return true;
  }

  // Allow integer to connect to number
  if (sourceType === "integer" && targetType === "number") {
    return true;
  }

  // Check for array/object special cases
  if (targetType === "object" && sourceType === "object") {
    return true;
  }

  if (targetType === "array" && sourceType === "array") {
    return true;
  }

  return false;
}

/**
 * Get all input handles from a block that are compatible with the given output type
 */
export function getCompatibleInputs(
  block: Block,
  outputType: string | undefined,
  includeDynamic: boolean = false,
): string[] {
  const compatibleInputs: string[] = [];

  if (!block.inputSchema?.properties) {
    return compatibleInputs;
  }

  for (const [key, schema] of Object.entries(block.inputSchema.properties)) {
    // Check for 'any' type first - it accepts everything
    // Cast to string to handle the any type comparison
    if ((schema.type as string) === "any" || outputType === "any") {
      compatibleInputs.push(key);
      continue;
    }

    // Arrays - non-list types can be connected to create new elements
    if (schema.type === "array") {
      // Direct array-to-array connection (not dynamic)
      if (outputType === "array") {
        compatibleInputs.push(key);
      }
      // For 'any' or undefined types, we'll handle them as user choice later
      else if (outputType === "any" || outputType === undefined) {
        compatibleInputs.push(key);
      }
      // Dynamic append for non-array types
      else if (includeDynamic && "items" in schema && schema.items) {
        const itemsType = schema.items.type;
        // Can append to array if item type matches OR if the array accepts 'any' or undefined
        if (
          (schema.items.type as string) === "any" ||
          itemsType === undefined ||
          areTypesCompatible(outputType, schema.items.type)
        ) {
          compatibleInputs.push(key);
        }
      }
    } else if (schema.type === "object") {
      // Check if this is a dictionary (has additionalProperties)
      const isDict =
        "additionalProperties" in schema && schema.additionalProperties;

      if (isDict) {
        // Direct dict-to-dict connection
        if (outputType === "object") {
          compatibleInputs.push(key);
        }
        // For 'any' or undefined types, we'll handle them as user choice later
        else if (outputType === "any" || outputType === undefined) {
          compatibleInputs.push(key);
        }
        // Dynamic key-value pair for non-dict types
        else if (includeDynamic) {
          const additionalType = getAdditionalPropertiesType(schema);
          if (
            additionalType === "any" ||
            additionalType === undefined ||
            areTypesCompatible(outputType, additionalType)
          ) {
            compatibleInputs.push(key);
          }
        }
      } else {
        // Regular object type (not a dynamic dict) - only match object to object
        if (areTypesCompatible(outputType, schema.type)) {
          compatibleInputs.push(key);
        }
      }
    } else {
      // Direct type compatibility for non-container types
      if (areTypesCompatible(outputType, schema.type)) {
        compatibleInputs.push(key);
      }
    }
  }

  return compatibleInputs;
}

/**
 * Get all output handles from a block that are compatible with the given input type
 */
export function getCompatibleOutputs(
  block: Block,
  inputType: string | undefined,
): string[] {
  const compatibleOutputs: string[] = [];

  if (!block.outputSchema?.properties) {
    return compatibleOutputs;
  }

  for (const [key, schema] of Object.entries(block.outputSchema.properties)) {
    // Check for 'any' type first - it accepts/provides everything
    // Cast to string to handle the any type comparison
    if ((schema.type as string) === "any" || inputType === "any") {
      compatibleOutputs.push(key);
      continue;
    }

    if (areTypesCompatible(schema.type, inputType)) {
      compatibleOutputs.push(key);
    }
  }

  return compatibleOutputs;
}

/**
 * Filter blocks to only those that have compatible connections
 */
export function filterBlocksByConnectionType(
  blocks: Block[],
  connectionType: "input" | "output",
  handleType: string | undefined,
  _handleKey?: string,
): Block[] {
  const filteredBlocks = blocks.filter((block) => {
    if (connectionType === "output") {
      // We're dragging from an output, so we need blocks with compatible inputs
      // Include dynamic connections (arrays and dicts)
      const compatibleInputs = getCompatibleInputs(block, handleType, true);
      const isCompatible = compatibleInputs.length > 0;
      return isCompatible;
    } else {
      // We're dragging from an input, so we need blocks with compatible outputs
      const compatibleOutputs = getCompatibleOutputs(block, handleType);
      const isCompatible = compatibleOutputs.length > 0;
      return isCompatible;
    }
  });

  return filteredBlocks;
}

/**
 * Check if a block can accept dict/array key additions
 */
export function canAddDynamicKey(
  schema: BlockIOSubSchema | undefined,
): boolean {
  if (!schema) return false;

  // Check for object with additionalProperties
  if (schema.type === "object" && "additionalProperties" in schema) {
    return true;
  }

  // Check for array type
  if (schema.type === "array") {
    return true;
  }

  return false;
}

/**
 * Get the type for additional properties in a dict/object schema
 */
export function getAdditionalPropertiesType(
  schema: BlockIOSubSchema | undefined,
): string | undefined {
  if (!schema || schema.type !== "object") {
    return undefined;
  }

  if ("additionalProperties" in schema && schema.additionalProperties) {
    return schema.additionalProperties.type;
  }

  return undefined;
}

/**
 * Get the items type for an array schema
 */
export function getArrayItemsType(
  schema: BlockIOSubSchema | undefined,
): string | undefined {
  if (!schema || schema.type !== "array") {
    return undefined;
  }

  if ("items" in schema && schema.items) {
    return schema.items.type;
  }

  return undefined;
}
