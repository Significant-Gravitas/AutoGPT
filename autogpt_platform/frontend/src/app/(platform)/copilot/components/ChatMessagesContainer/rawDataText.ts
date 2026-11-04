/** A reply that is nothing but a JSON object or array is a tool result the
 *  model echoed, not something written for the user: it folds under "Show
 *  work" instead of reading as the answer. A fenced block stays, since that
 *  is how the model shows JSON the user asked for. */
export function isRawDataText(text: string): boolean {
  const trimmed = text.trim();
  if (!/^[[{]/.test(trimmed) || !/[\]}]$/.test(trimmed)) return false;
  try {
    const value: unknown = JSON.parse(trimmed);
    return typeof value === "object" && value !== null;
  } catch {
    return false;
  }
}
