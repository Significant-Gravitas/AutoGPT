// Kept out of the "use client" sidebar module so server components can read it too.
// Not `sidebar_state`: automatic collapses wrote that one too, so its value is no choice.
export const SIDEBAR_COOKIE_NAME = "sidebar_open";

export function parseSidebarCookie(
  value: string | undefined,
): boolean | undefined {
  if (value === "true") return true;
  if (value === "false") return false;
  return undefined;
}
