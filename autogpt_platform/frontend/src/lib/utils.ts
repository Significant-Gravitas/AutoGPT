import { type ClassValue, clsx } from "clsx";
import _isEmpty from "lodash/isEmpty";
import { extendTailwindMerge } from "tailwind-merge";

import {
  BlockIOObjectSubSchema,
  BlockIORootSchema,
  BlockIOSubSchema,
  GraphInputSubSchema,
  GraphOutputSubSchema,
} from "@/lib/autogpt-server-api/types";

// Teach tailwind-merge the custom theme keys in src/app/globals.css, so a
// later `rounded-md` replaces `rounded-large` instead of both being kept and
// stylesheet order deciding which one wins.
const twMerge = extendTailwindMerge({
  extend: {
    theme: {
      radius: ["xsmall", "small", "medium", "large", "xlarge", "2xlarge"],
      shadow: ["subtle"],
    },
  },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Derived from https://stackoverflow.com/a/32922084 */
export function deepEquals(x: any, y: any): boolean {
  const ok = (obj: any) => Object.keys(obj).filter((key) => obj[key] !== null),
    tx = typeof x,
    ty = typeof y;

  const res =
    x && y && tx === ty && tx === "object"
      ? ok(x).length === ok(y).length &&
        ok(x).every((key) => deepEquals(x[key], y[key]))
      : x === y;
  return res;
}

/**
 * Extracts the effective type from a JSON schema, handling anyOf/oneOf/allOf wrappers.
 * Returns the first non-null type found in the schema structure.
 */
export function getEffectiveType(
  schema:
    | BlockIOSubSchema
    | GraphInputSubSchema
    | GraphOutputSubSchema
    | null
    | undefined,
): string | undefined {
  if (!schema) return undefined;

  // Direct type property
  if ("type" in schema && schema.type) {
    return String(schema.type);
  }

  // Handle allOf - typically a single-item wrapper
  if (
    "allOf" in schema &&
    Array.isArray(schema.allOf) &&
    schema.allOf.length > 0
  ) {
    return getEffectiveType(schema.allOf[0]);
  }

  // Handle anyOf - e.g. [{ type: "string" }, { type: "null" }]
  if ("anyOf" in schema && Array.isArray(schema.anyOf)) {
    for (const item of schema.anyOf) {
      if ("type" in item && item.type !== "null") {
        return String(item.type);
      }
    }
  }

  // Handle oneOf
  if ("oneOf" in schema && Array.isArray(schema.oneOf)) {
    for (const item of schema.oneOf) {
      if ("type" in item && item.type !== "null") {
        return String(item.type);
      }
    }
  }

  return undefined;
}

export function beautifyString(name: string): string {
  // Regular expression to identify places to split, considering acronyms
  const result = name
    .replace(/([a-z])([A-Z])/g, "$1 $2") // Add space before capital letters
    .replace(/([A-Z])([A-Z][a-z])/g, "$1 $2") // Add space between acronyms and next word
    .replace(/_/g, " ") // Replace underscores with spaces
    .replace(/\b\w/g, (char) => char.toUpperCase()); // Capitalize the first letter of each word

  return applyExceptions(result);
}

const exceptionMap: Record<string, string> = {
  "Auto GPT": "AutoGPT",
  Gpt: "GPT",
  Creds: "Credentials",
  Id: "ID",
  Openai: "OpenAI",
  Api: "API",
  Url: "URL",
  Http: "HTTP",
  Json: "JSON",
  Ai: "AI",
  "You Tube": "YouTube",
  "All Quiet": "AllQuiet",
  "Any Search": "AnySearch",
};

const applyExceptions = (str: string): string => {
  Object.keys(exceptionMap).forEach((key) => {
    const regex = new RegExp(`\\b${key}\\b`, "g");
    str = str.replace(regex, exceptionMap[key]);
  });
  return str;
};

export function exportAsJSONFile(obj: object, filename: string): void {
  // Create downloadable blob
  const jsonString = JSON.stringify(obj, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  // Trigger the browser to download the blob to a file
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  // Clean up
  URL.revokeObjectURL(url);
}

export function agentGraphExportFilename(
  graph: unknown,
  fallbackName = "agent",
): string {
  const { name, version } = (
    typeof graph === "object" && graph !== null ? graph : {}
  ) as { name?: unknown; version?: unknown };
  const rawName = typeof name === "string" && name.trim() ? name : fallbackName;
  const safeName =
    rawName
      .trim()
      .replace(/[\\/:*?"<>|\x00-\x1f]/g, "_")
      .replace(/[_ ]+$/, "") || "agent";
  return typeof version === "number"
    ? `${safeName}_v${version}.json`
    : `${safeName}.json`;
}

export function setNestedProperty(obj: any, path: string, value: any) {
  if (!obj || typeof obj !== "object") {
    throw new Error("Target must be a non-null object");
  }

  if (!path || typeof path !== "string") {
    throw new Error("Path must be a non-empty string");
  }

  // Split by both / and . to handle mixed separators, then filter empty strings
  const keys = path.split(/[\/.]/).filter((key) => key.length > 0);

  if (keys.length === 0) {
    throw new Error("Path must be a non-empty string");
  }

  // Validate keys for prototype pollution protection
  for (const key of keys) {
    if (key === "__proto__" || key === "constructor" || key === "prototype") {
      throw new Error(`Invalid property name: ${key}`);
    }
  }

  // Securely traverse and set nested properties
  // Use Object.prototype.hasOwnProperty.call() to safely check properties
  let current = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];

    // Use hasOwnProperty check to avoid prototype chain access
    if (!Object.prototype.hasOwnProperty.call(current, key)) {
      current[key] = {};
    } else if (typeof current[key] !== "object" || current[key] === null) {
      current[key] = {};
    }
    current = current[key];
  }

  // Set the final value using bracket notation with validated key
  // Since we've validated all keys, this is safe from prototype pollution
  const finalKey = keys[keys.length - 1];
  current[finalKey] = value;
}

export function pruneEmptyValues(
  obj: any,
  removeEmptyStrings: boolean = true,
): any {
  if (Array.isArray(obj)) {
    // If obj is an array, recursively check each element,
    // but element removal is avoided to prevent index changes.
    return obj.map((item) =>
      item === undefined || item === null
        ? ""
        : pruneEmptyValues(item, removeEmptyStrings),
    );
  } else if (typeof obj === "object" && obj !== null) {
    // If obj is an object, recursively remove empty strings and nulls from its properties
    for (const key in obj) {
      if (!obj.hasOwnProperty(key)) continue;

      const value = obj[key];
      if (
        value === null ||
        value === undefined ||
        (typeof value === "string" && value === "" && removeEmptyStrings)
      ) {
        delete obj[key];
      } else if (typeof value === "object") {
        obj[key] = pruneEmptyValues(value, removeEmptyStrings);
      }
    }
  }
  return obj;
}

export function fillObjectDefaultsFromSchema(
  obj: Record<any, any>,
  schema: BlockIORootSchema | BlockIOObjectSubSchema,
) {
  for (const key in schema.properties) {
    if (!schema.properties.hasOwnProperty(key)) continue;

    const propertySchema = schema.properties[key];

    if ("default" in propertySchema && propertySchema.default !== undefined) {
      // Apply simple default values
      obj[key] ??= propertySchema.default;
    } else if (
      propertySchema.type === "object" &&
      "properties" in propertySchema
    ) {
      // Recursively fill defaults for nested objects
      obj[key] = fillObjectDefaultsFromSchema(obj[key] ?? {}, propertySchema);
    } else if (propertySchema.type === "array") {
      obj[key] ??= [];
      // If the array items are objects, fill their defaults as well
      if (
        Array.isArray(obj[key]) &&
        propertySchema.items?.type === "object" &&
        "properties" in propertySchema.items
      ) {
        for (const item of obj[key]) {
          if (typeof item === "object" && item !== null) {
            fillObjectDefaultsFromSchema(item, propertySchema.items);
          }
        }
      }
    }
  }

  return obj;
}

export function hasNonNullNonObjectValue(obj: any): boolean {
  if (obj !== null && typeof obj === "object") {
    return Object.values(obj).some((value) => hasNonNullNonObjectValue(value));
  } else {
    return obj !== null && typeof obj !== "object";
  }
}

export function isEmpty(value: any): boolean {
  return (
    value === undefined ||
    value === "" ||
    (typeof value === "object" &&
      (value instanceof Date ? isNaN(value.getTime()) : _isEmpty(value))) ||
    (typeof value === "number" && isNaN(value))
  );
}

/** Validate YouTube URL */
export function validateYouTubeUrl(val: string): boolean {
  if (!val) return true;
  try {
    const url = new URL(val);
    const allowedHosts = [
      "youtube.com",
      "www.youtube.com",
      "youtu.be",
      "www.youtu.be",
    ];
    return allowedHosts.includes(url.hostname);
  } catch {
    return false;
  }
}

export function isValidUUID(value: string): boolean {
  const uuidRegex =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  return uuidRegex.test(value);
}

export function getApprovedMarketplaceUrl(args: {
  creatorUsername: string | null | undefined;
  slug: string | null | undefined;
  isApproved: boolean;
}): string | undefined {
  if (!args.isApproved || !args.creatorUsername || !args.slug) {
    return undefined;
  }
  return `/marketplace/agent/${encodeURIComponent(args.creatorUsername)}/${encodeURIComponent(args.slug)}`;
}

/**
 * True when `pathname` is `base` itself or one of its sub-routes. Tolerates the
 * `null` that `usePathname()` is typed to return.
 */
export function matchesRoute(
  pathname: string | null | undefined,
  base: string,
): boolean {
  if (!pathname) return false;
  return pathname === base || pathname.startsWith(`${base}/`);
}

export function getTypeColor(type: string | undefined): string {
  if (!type) return "#6b7280";
  return (
    {
      string: "#22c55e",
      number: "#3b82f6",
      integer: "#3b82f6",
      boolean: "#eab308",
      object: "#a855f7",
      array: "#6366f1",
      null: "#6b7280",
      any: "#6b7280",
    }[type] || "#6b7280"
  );
}
