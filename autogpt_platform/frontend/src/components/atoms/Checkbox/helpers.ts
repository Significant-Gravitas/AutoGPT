export const checkboxSizeClasses = {
  sm: "",
  md: "size-5 rounded-md [&_svg]:size-3.5",
} as const;

// Kobra's indicator only draws the check mark; the indeterminate dash is a
// pseudo-element so the vendored file stays untouched.
export const INDETERMINATE_CLASSES =
  "data-indeterminate:border-primary data-indeterminate:bg-primary data-indeterminate:text-primary-foreground data-indeterminate:[&_svg]:hidden data-indeterminate:before:absolute data-indeterminate:before:h-0.5 data-indeterminate:before:w-1/2 data-indeterminate:before:rounded-full data-indeterminate:before:bg-current";

export function getDescribedBy(
  ...ids: Array<string | undefined>
): string | undefined {
  const joined = ids.filter(Boolean).join(" ");
  return joined || undefined;
}
