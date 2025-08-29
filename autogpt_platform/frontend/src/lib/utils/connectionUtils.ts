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
  console.log("[areTypesCompatible] Checking compatibility:", {
    sourceType,
    targetType,
  });

  // Any type can connect to anything
  if (sourceType === "any" || targetType === "any") {
    console.log('[areTypesCompatible] ✅ Compatible: one side is "any"');
    return true;
  }

  // If either is undefined, allow connection (legacy behavior)
  if (!sourceType || !targetType) {
    console.log(
      "[areTypesCompatible] ✅ Compatible: undefined type (legacy behavior)",
    );
    return true;
  }

  // Direct type match
  if (sourceType === targetType) {
    console.log("[areTypesCompatible] ✅ Compatible: exact type match");
    return true;
  }

  // Allow integer to connect to number
  if (sourceType === "integer" && targetType === "number") {
    console.log(
      "[areTypesCompatible] ✅ Compatible: integer to number conversion",
    );
    return true;
  }

  // Check for array/object special cases
  if (targetType === "object" && sourceType === "object") {
    console.log("[areTypesCompatible] ✅ Compatible: both are objects");
    return true;
  }

  if (targetType === "array" && sourceType === "array") {
    console.log("[areTypesCompatible] ✅ Compatible: both are arrays");
    return true;
  }

  console.log("[areTypesCompatible] ❌ Not compatible");
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
  console.log("[getCompatibleInputs] Starting check:", {
    blockName: block.name,
    outputType,
    includeDynamic,
    inputSchemaKeys: block.inputSchema?.properties
      ? Object.keys(block.inputSchema.properties)
      : [],
  });

  const compatibleInputs: string[] = [];

  if (!block.inputSchema?.properties) {
    console.log("[getCompatibleInputs] No input schema properties found");
    return compatibleInputs;
  }

  for (const [key, schema] of Object.entries(block.inputSchema.properties)) {
    console.log(`[getCompatibleInputs] Checking input "${key}":`, {
      type: schema.type,
      hasItems: "items" in schema,
      hasAdditionalProperties: "additionalProperties" in schema,
    });

    // Check for 'any' type first - it accepts everything
    // Cast to string to handle the any type comparison
    if ((schema.type as string) === "any" || outputType === "any") {
      console.log(`[getCompatibleInputs] ✅ "${key}" accepts any type`);
      compatibleInputs.push(key);
      continue;
    }

    // Arrays - non-list types can be connected to create new elements
    if (schema.type === "array") {
      // Direct array-to-array connection (not dynamic)
      if (outputType === "array") {
        console.log(
          `[getCompatibleInputs] ✅ Array-to-array direct connection for "${key}"`,
        );
        compatibleInputs.push(key);
      }
      // For 'any' or undefined types, we'll handle them as user choice later
      else if (outputType === "any" || outputType === undefined) {
        console.log(
          `[getCompatibleInputs] ✅ Any/undefined to array for "${key}" - will need user choice`,
        );
        compatibleInputs.push(key);
      }
      // Dynamic append for non-array types
      else if (includeDynamic && "items" in schema && schema.items) {
        const itemsType = schema.items.type;
        console.log(
          `[getCompatibleInputs] Checking array "${key}" for dynamic append with items type:`,
          itemsType,
        );
        // Can append to array if item type matches OR if the array accepts 'any' or undefined
        if (
          (schema.items.type as string) === "any" ||
          itemsType === undefined ||
          areTypesCompatible(outputType, schema.items.type)
        ) {
          console.log(
            `[getCompatibleInputs] ✅ Can dynamically append to array "${key}"`,
          );
          compatibleInputs.push(key);
        } else {
          console.log(
            `[getCompatibleInputs] ❌ Cannot append to array "${key}" - type mismatch`,
          );
        }
      } else {
        console.log(
          `[getCompatibleInputs] Array "${key}" skipped - not matching criteria`,
        );
      }
    } else if (schema.type === "object") {
      // Check if this is a dictionary (has additionalProperties)
      const isDict =
        "additionalProperties" in schema && schema.additionalProperties;

      if (isDict) {
        // Direct dict-to-dict connection
        if (outputType === "object") {
          console.log(
            `[getCompatibleInputs] ✅ Object-to-dict direct connection for "${key}"`,
          );
          compatibleInputs.push(key);
        }
        // For 'any' or undefined types, we'll handle them as user choice later
        else if (outputType === "any" || outputType === undefined) {
          console.log(
            `[getCompatibleInputs] ✅ Any/undefined to dict for "${key}" - will need user choice`,
          );
          compatibleInputs.push(key);
        }
        // Dynamic key-value pair for non-dict types
        else if (includeDynamic) {
          const additionalType = getAdditionalPropertiesType(schema);
          console.log(
            `[getCompatibleInputs] Checking dict "${key}" for dynamic key-value with value type:`,
            additionalType,
          );
          if (
            additionalType === "any" ||
            additionalType === undefined ||
            areTypesCompatible(outputType, additionalType)
          ) {
            console.log(
              `[getCompatibleInputs] ✅ Can dynamically add to dict "${key}"`,
            );
            compatibleInputs.push(key);
          } else {
            console.log(
              `[getCompatibleInputs] ❌ Cannot add to dict "${key}" - type mismatch`,
            );
          }
        } else {
          console.log(
            `[getCompatibleInputs] Dict "${key}" skipped - not matching criteria`,
          );
        }
      } else {
        // Regular object type (not a dynamic dict) - only match object to object
        if (areTypesCompatible(outputType, schema.type)) {
          console.log(`[getCompatibleInputs] ✅ Object "${key}" is compatible`);
          compatibleInputs.push(key);
        } else {
          console.log(
            `[getCompatibleInputs] ❌ Object "${key}" is not compatible`,
          );
        }
      }
    } else {
      // Direct type compatibility for non-container types
      if (areTypesCompatible(outputType, schema.type)) {
        console.log(`[getCompatibleInputs] ✅ Direct type match for "${key}"`);
        compatibleInputs.push(key);
      } else {
        console.log(
          `[getCompatibleInputs] ❌ Type mismatch for "${key}": ${outputType} vs ${schema.type}`,
        );
      }
    }
  }

  console.log(
    "[getCompatibleInputs] Final compatible inputs:",
    compatibleInputs,
  );
  return compatibleInputs;
}

/**
 * Get all output handles from a block that are compatible with the given input type
 */
export function getCompatibleOutputs(
  block: Block,
  inputType: string | undefined,
): string[] {
  console.log("[getCompatibleOutputs] Starting check:", {
    blockName: block.name,
    inputType,
    outputSchemaKeys: block.outputSchema?.properties
      ? Object.keys(block.outputSchema.properties)
      : [],
  });

  const compatibleOutputs: string[] = [];

  if (!block.outputSchema?.properties) {
    console.log("[getCompatibleOutputs] No output schema properties found");
    return compatibleOutputs;
  }

  for (const [key, schema] of Object.entries(block.outputSchema.properties)) {
    console.log(`[getCompatibleOutputs] Checking output "${key}":`, {
      type: schema.type,
      schema: schema,
      hasType: "type" in schema,
    });

    // Check for 'any' type first - it accepts/provides everything
    // Cast to string to handle the any type comparison
    if ((schema.type as string) === "any" || inputType === "any") {
      console.log(
        `[getCompatibleOutputs] ✅ "${key}" is any type or connecting to any`,
      );
      compatibleOutputs.push(key);
      continue;
    }

    if (areTypesCompatible(schema.type, inputType)) {
      console.log(`[getCompatibleOutputs] ✅ "${key}" is compatible`);
      compatibleOutputs.push(key);
    } else {
      console.log(
        `[getCompatibleOutputs] ❌ "${key}" is not compatible: ${schema.type} vs ${inputType}`,
      );
    }
  }

  console.log(
    "[getCompatibleOutputs] Final compatible outputs:",
    compatibleOutputs,
  );
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
  console.log("[filterBlocksByConnectionType] Starting filter:", {
    totalBlocks: blocks.length,
    connectionType,
    handleType,
  });

  const filteredBlocks = blocks.filter((block) => {
    console.log(
      `[filterBlocksByConnectionType] Checking block "${block.name}"`,
    );

    if (connectionType === "output") {
      // We're dragging from an output, so we need blocks with compatible inputs
      // Include dynamic connections (arrays and dicts)
      const compatibleInputs = getCompatibleInputs(block, handleType, true);
      const isCompatible = compatibleInputs.length > 0;
      console.log(
        `[filterBlocksByConnectionType] Block "${block.name}" has ${compatibleInputs.length} compatible inputs: ${isCompatible ? "✅" : "❌"}`,
      );
      return isCompatible;
    } else {
      // We're dragging from an input, so we need blocks with compatible outputs
      const compatibleOutputs = getCompatibleOutputs(block, handleType);
      const isCompatible = compatibleOutputs.length > 0;
      console.log(
        `[filterBlocksByConnectionType] Block "${block.name}" has ${compatibleOutputs.length} compatible outputs: ${isCompatible ? "✅" : "❌"}`,
      );
      return isCompatible;
    }
  });

  console.log("[filterBlocksByConnectionType] Final filtered blocks:", {
    totalFiltered: filteredBlocks.length,
    blockNames: filteredBlocks.map((b) => b.name),
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
