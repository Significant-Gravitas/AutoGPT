/**
 * Shared type definitions for chat-related data structures.
 * These types provide type-safe alternatives to Record<string, any>.
 */

/**
 * Represents a valid JSON value that can be used in tool arguments or results.
 * This is a recursive type that allows for nested objects and arrays.
 */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * Represents tool arguments passed to a tool call.
 * Can be a simple object with string keys and JSON values.
 */
export interface ToolArguments {
  [key: string]: JsonValue;
}

/**
 * Represents the result returned from a tool execution.
 * Can be either a string or a structured object with JSON values.
 */
export type ToolResult = string | { [key: string]: JsonValue };
