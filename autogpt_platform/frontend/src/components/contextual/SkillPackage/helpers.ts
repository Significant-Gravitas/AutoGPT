const UNITS = ["B", "KB", "MB"];

/** Deliberately not `toLocaleString`: this machine's Dutch locale renders
 *  1,024 as 1.024, and a package's sizes are compared as rendered text. */
export function formatFileSize(bytes: number): string {
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < UNITS.length - 1) {
    size /= 1024;
    unit += 1;
  }
  // Non-breaking: a size must never wrap between its number and unit.
  return `${unit === 0 ? size : size.toFixed(1)}\u00a0${UNITS[unit]}`;
}

/** The package file a body link points at, or `null` when the link is not one
 *  — an absolute URL, an anchor, or a path this package does not ship. */
export function resolvePackagePath(
  href: string | undefined,
  paths: string[],
): string | null {
  if (!href) return null;
  const isAbsolute =
    /^[a-z][a-z0-9+.-]*:/i.test(href) ||
    href.startsWith("//") ||
    href.startsWith("/");
  if (isAbsolute || href.startsWith("#")) return null;

  const bare = href.split(/[?#]/)[0].replace(/^\.\//, "");
  if (!bare) return null;
  const decoded = decodePath(bare);
  return paths.includes(decoded) ? decoded : null;
}

function decodePath(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}
