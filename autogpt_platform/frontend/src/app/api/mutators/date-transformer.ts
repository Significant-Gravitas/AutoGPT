/**
 * Date transformation utility for converting ISO date strings to Date objects
 * in API responses. This handles the conversion recursively for nested objects.
 */

// ISO date regex pattern to match strings that look like UTC ISO dates.
// The trailing `Z` is required: a zone-less string ("2024-03-10T09:30:00")
// would otherwise be parsed as browser-local time and shift on re-serialize.
const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/;

// Keys holding user-supplied graph data. Their values are copied verbatim so
// "Run again" resubmits exactly what the user entered.
const USER_PAYLOAD_KEYS = new Set([
  "inputs",
  "outputs",
  "input_data",
  "output_data",
  "credential_inputs",
  "nodes_input_masks",
]);

/**
 * Validates if a string is a valid ISO date and can be parsed
 */
function isValidISODate(dateString: string): boolean {
  if (!ISO_DATE_REGEX.test(dateString)) {
    return false;
  }

  const date = new Date(dateString);
  return !isNaN(date.getTime());
}

/**
 * Recursively transforms ISO date strings to Date objects in an object or array
 * @param obj - The object or array to transform
 * @returns The transformed object with Date objects
 */
export function transformDates<T>(obj: T): T {
  if (typeof obj !== "object" || obj === null) return obj;

  // Handle arrays
  if (Array.isArray(obj)) {
    return obj.map(transformDates) as T;
  }

  // Handle objects
  const transformed = {} as T;

  for (const [key, value] of Object.entries(obj)) {
    if (USER_PAYLOAD_KEYS.has(key)) {
      // User data, not server timestamps: leave untouched
      (transformed as any)[key] = value;
    } else if (typeof value === "string" && isValidISODate(value)) {
      // Convert ISO date string to Date object
      (transformed as any)[key] = new Date(value);
    } else if (typeof value === "object") {
      // Recursively transform nested objects/arrays
      (transformed as any)[key] = transformDates(value);
    } else {
      // Keep primitive values as-is
      (transformed as any)[key] = value;
    }
  }

  return transformed;
}
