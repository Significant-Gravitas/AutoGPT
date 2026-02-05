/**
 * Constants for the chat system.
 *
 * Centralizes magic strings and values used across chat components.
 */

// LocalStorage keys
export const STORAGE_KEY_ACTIVE_TASKS = "chat_active_tasks";

// Redis Stream IDs
export const INITIAL_MESSAGE_ID = "0";
export const INITIAL_STREAM_ID = "0-0";

// TTL values (in milliseconds)
export const COMPLETED_STREAM_TTL_MS = 5 * 60 * 1000; // 5 minutes
export const ACTIVE_TASK_TTL_MS = 60 * 60 * 1000; // 1 hour
