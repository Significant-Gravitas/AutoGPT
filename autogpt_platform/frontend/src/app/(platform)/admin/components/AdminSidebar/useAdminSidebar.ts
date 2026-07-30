import { usePathname } from "next/navigation";
import { adminNavItems } from "./helpers";

export function useAdminSidebar() {
  const pathname = usePathname();

  const items = adminNavItems.map((item) => ({
    ...item,
    isActive: pathname === item.href || pathname.startsWith(`${item.href}/`),
  }));

  return { items };
}
