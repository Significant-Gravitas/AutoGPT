import { matchesRoute } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { getAdminNavItems } from "./helpers";

export function useAdminSidebar() {
  const pathname = usePathname();

  const items = getAdminNavItems().map((item) => ({
    ...item,
    isActive: matchesRoute(pathname, item.href),
  }));

  return { items };
}
