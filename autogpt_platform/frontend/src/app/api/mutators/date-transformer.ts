/**
 * Date transformation utility for converting ISO date strings to Date objects
 * in API responses. This handles the conversion recursively for nested objects.
 */

// ISO date regex pattern to match strings that look like ISO dates
const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$/;

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
    if (typeof value === "string" && isValidISODate(value)) {
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
