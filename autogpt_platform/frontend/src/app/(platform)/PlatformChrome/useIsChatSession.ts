import { usePathname, useSearchParams } from "next/navigation";

import { isCopilotPathname } from "./helpers";

// An open chat carries the sidebar and workspace-files buttons in its thread
// chip. Kept out of usePlatformChrome on purpose: reading the search params
// re-renders the caller on every URL-param write (copilot `sessionId`, builder
// `flowVersion`, library `activeItem`), so only the header leaf that needs
// the answer subscribes to them.
export function useIsChatSession() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  return isCopilotPathname(pathname) && Boolean(searchParams.get("sessionId"));
}
