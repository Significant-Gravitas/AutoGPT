import { usePathname } from "next/navigation";
import { settingsNavItems } from "./helpers";

export function useSettingsSidebar() {
  const pathname = usePathname();

  const items = settingsNavItems.map((item) => ({
    ...item,
    isActive:
      pathname === item.href ||
      (item.href !== "/settings" && pathname.startsWith(`${item.href}/`)),
  }));

  return { items };
}
