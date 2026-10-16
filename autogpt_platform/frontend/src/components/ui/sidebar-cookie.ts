// Kept out of the "use client" sidebar module so server components can read it too.
export const SIDEBAR_COOKIE_NAME = "sidebar_state";

export function parseSidebarCookie(
  value: string | undefined,
): boolean | undefined {
  if (value === "true") return true;
  if (value === "false") return false;
  return undefined;
}
