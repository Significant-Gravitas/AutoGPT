import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

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
  return (
    x &&
    y &&
    tx === ty &&
    (tx === "object"
      ? ok(x).length === ok(y).length &&
        ok(x).every((key) => deepEquals(x[key], y[key]))
      : x === y)
  );
}

/** Get tailwind text color class from type name */
export function getTypeTextColor(type: string | null): string {
  if (type === null) return "bg-gray-500";
  return (
    {
      string: "text-green-500",
      number: "text-blue-500",
      boolean: "text-yellow-500",
      object: "text-purple-500",
      array: "text-indigo-500",
      null: "text-gray-500",
      "": "text-gray-500",
    }[type] || "text-gray-500"
  );
}

/** Get tailwind bg color class from type name */
export function getTypeBgColor(type: string | null): string {
  if (type === null) return "bg-gray-500";
  return (
    {
      string: "bg-green-500",
      number: "bg-blue-500",
      boolean: "bg-yellow-500",
      object: "bg-purple-500",
      array: "bg-indigo-500",
      null: "bg-gray-500",
      "": "bg-gray-500",
    }[type] || "bg-gray-500"
  );
}

export function getTypeColor(type: string | null): string {
  if (type === null) return "bg-gray-500";
  return (
    {
      string: "#22c55e",
      number: "#3b82f6",
      boolean: "#eab308",
      object: "#a855f7",
      array: "#6366f1",
      null: "#6b7280",
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

export function setNestedProperty(obj: any, path: string, value: any) {
  const keys = path.split(/[\/.]/); // Split by / or .
  let current = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!current[key] || typeof current[key] !== "object") {
      current[key] = {};
    }
    current = current[key];
  }

  current[keys[keys.length - 1]] = value;
}

export function removeEmptyStringsAndNulls(obj: any): any {
  if (Array.isArray(obj)) {
    // If obj is an array, recursively remove empty strings and nulls from its elements
    return obj
      .map((item) => removeEmptyStringsAndNulls(item))
      .filter(
        (item) =>
          item !== null && (typeof item !== "string" || item.trim() !== ""),
      );
  } else if (typeof obj === "object" && obj !== null) {
    // If obj is an object, recursively remove empty strings and nulls from its properties
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const value = obj[key];
        if (
          value === null ||
          (typeof value === "string" && value.trim() === "")
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
