import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { Category } from "./autogpt-server-api/types";
import { NodeDimension } from "@/components/Flow";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Derived from https://stackoverflow.com/a/7616484 */
export function hashString(str: string): number {
  let hash = 0,
    chr: number;
  if (str.length === 0) return hash;
  for (let i = 0; i < str.length; i++) {
    chr = str.charCodeAt(i);
    hash = (hash << 5) - hash + chr;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
}

/** Derived from https://stackoverflow.com/a/32922084 */
export function deepEquals(x: any, y: any): boolean {
  const ok = Object.keys,
    tx = typeof x,
    ty = typeof y;

  const res =
    x && y && tx === ty && tx === "object"
      ? ok(x).length === ok(y).length &&
        ok(x).every((key) => deepEquals(x[key], y[key]))
      : x === y;
  return res;
}

/** Get tailwind text color class from type name */
export function getTypeTextColor(type: string | null): string {
  if (type === null) return "text-gray-500";
  return (
    {
      string: "text-green-500",
      number: "text-blue-500",
      integer: "text-blue-500",
      boolean: "text-yellow-500",
      object: "text-purple-500",
      array: "text-indigo-500",
      null: "text-gray-500",
      any: "text-gray-500",
      "": "text-gray-500",
    }[type] || "text-gray-500"
  );
}

/** Get tailwind bg color class from type name */
export function getTypeBgColor(type: string | null): string {
  if (type === null) return "border-gray-500";
  return (
    {
      string: "border-green-500",
      number: "border-blue-500",
      integer: "border-blue-500",
      boolean: "border-yellow-500",
      object: "border-purple-500",
      array: "border-indigo-500",
      null: "border-gray-500",
      any: "border-gray-500",
      "": "border-gray-500",
    }[type] || "border-gray-500"
  );
}

export function getTypeColor(type: string | null): string {
  if (type === null) return "#6b7280";
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
      "": "#6b7280",
    }[type] || "#6b7280"
  );
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
};

const applyExceptions = (str: string): string => {
  Object.keys(exceptionMap).forEach((key) => {
    const regex = new RegExp(`\\b${key}\\b`, "g");
    str = str.replace(regex, exceptionMap[key]);
  });
  return str;
};

/** Recursively remove all "credentials" properties from exported JSON files */
export function removeCredentials(obj: any) {
  if (obj && typeof obj === "object") {
    if (Array.isArray(obj)) {
      obj.forEach((item) => removeCredentials(item));
    } else {
      delete obj.credentials;
      Object.values(obj).forEach((value) => removeCredentials(value));
    }
  }
  return obj;
}

export function exportAsJSONFile(obj: object, filename: string): void {
  // Deep clone the object to avoid modifying the original
  const sanitizedObj = JSON.parse(JSON.stringify(obj));

  // Sanitize the object
  removeCredentials(sanitizedObj);

  // Create downloadable blob
  const jsonString = JSON.stringify(sanitizedObj, null, 2);
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

export function setNestedProperty(obj: any, path: string, value: any) {
  if (!obj || typeof obj !== "object") {
    throw new Error("Target must be a non-null object");
  }

  if (!path || typeof path !== "string") {
    throw new Error("Path must be a non-empty string");
  }

  const keys = path.split(/[\/.]/);

  for (const key of keys) {
    if (
      !key ||
      key === "__proto__" ||
      key === "constructor" ||
      key === "prototype"
    ) {
      throw new Error(`Invalid property name: ${key}`);
    }
  }

  let current = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!current.hasOwnProperty(key)) {
      current[key] = {};
    } else if (typeof current[key] !== "object" || current[key] === null) {
      current[key] = {};
    }
    current = current[key];
  }

  current[keys[keys.length - 1]] = value;
}

export function removeEmptyStringsAndNulls(obj: any): any {
  if (Array.isArray(obj)) {
    // If obj is an array, recursively check each element,
    // but element removal is avoided to prevent index changes.
    return obj.map((item) =>
      item === undefined || item === null
        ? ""
        : removeEmptyStringsAndNulls(item),
    );
  } else if (typeof obj === "object" && obj !== null) {
    // If obj is an object, recursively remove empty strings and nulls from its properties
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const value = obj[key];
        if (
          value === null ||
          value === undefined ||
          (typeof value === "string" && value === "")
        ) {
          delete obj[key];
        } else {
          obj[key] = removeEmptyStringsAndNulls(value);
        }
      }
    }
  }
  return obj;
}

export const categoryColorMap: Record<string, string> = {
  AI: "bg-orange-300 dark:bg-orange-700",
  SOCIAL: "bg-yellow-300 dark:bg-yellow-700",
  TEXT: "bg-green-300 dark:bg-green-700",
  SEARCH: "bg-blue-300 dark:bg-blue-700",
  BASIC: "bg-purple-300 dark:bg-purple-700",
  INPUT: "bg-cyan-300 dark:bg-cyan-700",
  OUTPUT: "bg-red-300 dark:bg-red-700",
  LOGIC: "bg-teal-300 dark:bg-teal-700",
  DEVELOPER_TOOLS: "bg-fuchsia-300 dark:bg-fuchsia-700",
  AGENT: "bg-lime-300 dark:bg-lime-700",
};

export function getPrimaryCategoryColor(categories: Category[]): string {
  if (categories.length === 0) {
    return "bg-gray-300 dark:bg-slate-700";
  }
  return (
    categoryColorMap[categories[0].category] || "bg-gray-300 dark:bg-slate-700"
  );
}

export function filterBlocksByType<T>(
  blocks: T[],
  predicate: (block: T) => boolean,
): T[] {
  return blocks.filter(predicate);
}

export enum BehaveAs {
  CLOUD = "CLOUD",
  LOCAL = "LOCAL",
}

export function getBehaveAs(): BehaveAs {
  return process.env.NEXT_PUBLIC_BEHAVE_AS === "CLOUD"
    ? BehaveAs.CLOUD
    : BehaveAs.LOCAL;
}

function rectanglesOverlap(
  rect1: { x: number; y: number; width: number; height?: number },
  rect2: { x: number; y: number; width: number; height?: number },
): boolean {
  const x1 = rect1.x,
    y1 = rect1.y,
    w1 = rect1.width,
    h1 = rect1.height ?? 100;
  const x2 = rect2.x,
    y2 = rect2.y,
    w2 = rect2.width,
    h2 = rect2.height ?? 100;

  // Check if the rectangles do not overlap
  return !(x1 + w1 <= x2 || x1 >= x2 + w2 || y1 + h1 <= y2 || y1 >= y2 + h2);
}

export function findNewlyAddedBlockCoordinates(
  nodeDimensions: NodeDimension,
  newWidth: number,
  margin: number,
  zoom: number,
) {
  const nodeDimensionArray = Object.values(nodeDimensions);

  for (let i = nodeDimensionArray.length - 1; i >= 0; i--) {
    const lastNode = nodeDimensionArray[i];
    const lastNodeHeight = lastNode.height ?? 100;

    // Right of the last node
    let newX = lastNode.x + lastNode.width + margin;
    let newY = lastNode.y;
    let newRect = { x: newX, y: newY, width: newWidth, height: 100 / zoom };

    const collisionRight = nodeDimensionArray.some((node) =>
      rectanglesOverlap(newRect, node),
    );

    if (!collisionRight) {
      return { x: newX, y: newY };
    }

    // Left of the last node
    newX = lastNode.x - newWidth - margin;
    newRect = { x: newX, y: newY, width: newWidth, height: 100 / zoom };

    const collisionLeft = nodeDimensionArray.some((node) =>
      rectanglesOverlap(newRect, node),
    );

    if (!collisionLeft) {
      return { x: newX, y: newY };
    }

    // Below the last node
    newX = lastNode.x;
    newY = lastNode.y + lastNodeHeight + margin;
    newRect = { x: newX, y: newY, width: newWidth, height: 100 / zoom };

    const collisionBelow = nodeDimensionArray.some((node) =>
      rectanglesOverlap(newRect, node),
    );

    if (!collisionBelow) {
      return { x: newX, y: newY };
    }
  }

  // Default position if no space is found
  return {
    x: 0,
    y: 0,
  };
}

export function hasNonNullNonObjectValue(obj: any): boolean {
  if (obj !== null && typeof obj === "object") {
    return Object.values(obj).some((value) => hasNonNullNonObjectValue(value));
  } else {
    return obj !== null && typeof obj !== "object";
  }
}

type ParsedKey = { key: string; index?: number };

export function parseKeys(key: string): ParsedKey[] {
  const splits = key.split(/_@_|_#_|_\$_|\./);
  const keys: ParsedKey[] = [];
  let currentKey: string | null = null;

  splits.forEach((split) => {
    const isInteger = /^\d+$/.test(split);
    if (!isInteger) {
      if (currentKey !== null) {
        keys.push({ key: currentKey });
      }
      currentKey = split;
    } else {
      if (currentKey !== null) {
        keys.push({ key: currentKey, index: parseInt(split, 10) });
        currentKey = null;
      } else {
        throw new Error("Invalid key format: array index without a key");
      }
    }
  });

  if (currentKey !== null) {
    keys.push({ key: currentKey });
  }

  return keys;
}

/**
 * Get the value of a nested key in an object, handles arrays and objects.
 */
export function getValue(key: string, value: any) {
  const keys = parseKeys(key);
  return keys.reduce((acc, k) => {
    if (acc === undefined) return undefined;
    if (k.index !== undefined) {
      return Array.isArray(acc[k.key]) ? acc[k.key][k.index] : undefined;
    }
    return acc[k.key];
  }, value);
}
