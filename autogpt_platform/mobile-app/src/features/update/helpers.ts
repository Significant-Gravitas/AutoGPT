/** A version's numbers, "1.2.3" → [1, 2, 3]. Null when it isn't one. */
function parseVersion(version: string) {
  // Anything after a hyphen or plus ("1.2.0-rc.1", "1.2.0+45") is ignored.
  const core = version.trim().split(/[-+]/)[0];
  if (!/^\d+(\.\d+){0,2}$/.test(core)) return null;
  const parts = core.split(".").map(Number);
  while (parts.length < 3) parts.push(0);
  return parts;
}

/**
 * Whether this app is older than the oldest version the server still lets
 * in. Unknown or unreadable versions let the app in: a typo in a setting
 * must not lock everyone out.
 */
export function isBelowMinimum(current: string | null, minimum: string | null) {
  if (!current || !minimum) return false;
  const have = parseVersion(current);
  const need = parseVersion(minimum);
  if (!have || !need) return false;
  for (let index = 0; index < 3; index += 1) {
    if (have[index] !== need[index]) return have[index] < need[index];
  }
  return false;
}

export type UpdateGate = "checking" | "ok" | "required";

/** The gate from the check: an error or a timeout lets the app in. */
export function updateGate(status: "pending" | "error" | "success", minimum: string | null, current: string | null): UpdateGate {
  if (status === "pending") return "checking";
  if (status === "error") return "ok";
  return isBelowMinimum(current, minimum) ? "required" : "ok";
}

/** Where to update the app: its Play page, or its App Store page once the store ID is set. */
export function storeURL(platform: string, androidPackage: string | null, appStoreID: string | null) {
  if (platform === "android" && androidPackage) {
    return `market://details?id=${androidPackage}`;
  }
  if (platform === "ios" && appStoreID) {
    return `itms-apps://apps.apple.com/app/id${appStoreID}`;
  }
  return null;
}
