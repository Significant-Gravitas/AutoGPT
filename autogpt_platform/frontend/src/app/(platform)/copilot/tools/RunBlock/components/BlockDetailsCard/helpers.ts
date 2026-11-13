// Mirrors `_derived_title` in backend/copilot/tools/run_block.py, which drops a
// title this reproduces — drift here changes what the card displays.
export function deriveFieldTitle(propertyName: string) {
  return propertyName
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
