import { router } from "expo-router";

const OTTO = "otto";

/**
 * Opens Home with a recipient picked: an expert's ID, or null for Otto. `at`
 * makes every call a new request, so Home re-picks even when the recipient is
 * the same as last time and the user has changed it on Home since.
 */
export function openHome(expertID: string | null) {
  router.navigate({
    pathname: "/",
    params: { recipient: expertID ?? OTTO, at: String(Date.now()) },
  });
}

/** The expert ID a Home link asks for, or null for Otto. */
export function recipientFromParam(value: string | string[] | undefined) {
  const recipient = Array.isArray(value) ? value[0] : value;
  return recipient && recipient !== OTTO ? recipient : null;
}
