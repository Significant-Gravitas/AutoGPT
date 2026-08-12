import { matchesRoute } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { adminNavItems } from "./helpers";

export function useAdminSidebar() {
  const pathname = usePathname();

  const items = adminNavItems.map((item) => ({
    ...item,
    isActive: matchesRoute(pathname, item.href),
  }));

  return { items };
}
