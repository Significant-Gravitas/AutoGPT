import { matchesRoute } from "@/lib/utils";

export function isCopilotPathname(pathname: string) {
  return matchesRoute(pathname, "/home") || matchesRoute(pathname, "/copilot");
}
