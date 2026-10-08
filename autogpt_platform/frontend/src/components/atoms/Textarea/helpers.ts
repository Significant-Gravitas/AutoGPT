export const textareaSizeClasses = {
  sm: "px-3 py-2",
  md: "px-4 py-2.5",
} as const;

const textareaPaddingY = {
  sm: "1rem",
  md: "1.25rem",
} as const;

// Kobra's textarea sizes itself to its content (`field-sizing: content`), which
// ignores `rows`. Translate `rows` into the minimum height it used to set.
export function rowsMinHeight(
  rows: number,
  size: keyof typeof textareaPaddingY = "sm",
): string {
  return `calc(${rows}lh + ${textareaPaddingY[size]} + 2px)`;
}

export function getDescribedBy(
  ...ids: Array<string | undefined>
): string | undefined {
  const joined = ids.filter(Boolean).join(" ");
  return joined || undefined;
}

export function getLength(value: unknown): number {
  if (typeof value === "string") return value.length;
  if (typeof value === "number") return String(value).length;
  return 0;
}
