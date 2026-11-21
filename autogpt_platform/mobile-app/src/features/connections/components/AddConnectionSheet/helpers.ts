import { hostFromURL, normalizeServerURL } from "../../helpers";
import { matchesQuery, type AccountRow, type ConnectableService } from "../../listHelpers";

/** The row key of "MCP server", which asks for an address first. */
export const SERVER_ROW = "mcp:by-url";

const SERVER_TERMS = ["MCP server", "Any server by URL"];

interface Args {
  services: ConnectableService[];
  /** Accounts the expert could be allowed to use; empty in a chat with Otto. */
  accounts: AccountRow[];
  query: string;
}

/** What the sheet lists for a search: accounts first, then new services. */
export function sheetLists({ services, accounts, query }: Args) {
  const shownAccounts = accounts.filter((row) => matchesQuery([row.name, row.detail], query));
  const shownServices = services.filter((service) =>
    matchesQuery([service.name, service.description, ...service.providers], query),
  );
  const showServer = matchesQuery(SERVER_TERMS, query);
  return {
    accounts: shownAccounts,
    services: shownServices,
    showServer,
    isEmpty: shownAccounts.length === 0 && shownServices.length === 0 && !showServer,
  };
}

/** The address to sign in to, or null until it reads as an http(s) URL with a host. */
export function serverTarget(text: string) {
  const url = text.trim();
  // The host must come first: "https://:8080" has none.
  if (!/^https?:\/\/[^\s/?#:]+/i.test(url) || /\s/.test(url)) return null;
  const host = hostFromURL(url);
  return host ? { serverURL: normalizeServerURL(url), providerName: host } : null;
}
