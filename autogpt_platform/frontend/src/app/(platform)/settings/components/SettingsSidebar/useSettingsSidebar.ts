import { usePathname } from "next/navigation";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { settingsNavItems } from "./helpers";

export function useSettingsSidebar() {
  const pathname = usePathname();
  const isMemoryEnabled = useGetFlag(Flag.GRAPHITI_MEMORY);

  const items = settingsNavItems
    .filter((item) => item.flag !== Flag.GRAPHITI_MEMORY || isMemoryEnabled)
    .map((item) => ({
      ...item,
      isActive:
        pathname === item.href ||
        (item.href !== "/settings" && pathname.startsWith(`${item.href}/`)),
    }));

  return { items };
}
