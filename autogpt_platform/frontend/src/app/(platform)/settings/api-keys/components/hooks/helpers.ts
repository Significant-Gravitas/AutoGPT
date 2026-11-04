import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";

// Keys act only in your personal organization's default team for now. Sent
// with no organization or team header, a request is answered for exactly that
// scope, whichever organization is active in the app.
export function getPersonalScopeRequestInit(): RequestInit {
  return getTenantRequestInit(null, null);
}
