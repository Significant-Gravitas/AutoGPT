import { usePathname } from "next/navigation";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { settingsNavItems } from "./helpers";

export function useSettingsSidebar() {
  const pathname = usePathname();
  const isMemoryEnabled = useGetFlag(Flag.GRAPHITI_MEMORY);
  const isOrgSettingsEnabled = useGetFlag(Flag.SHOW_ORG_SETTINGS);
  const enabledFlags = new Set<Flag>();
  if (isMemoryEnabled) enabledFlags.add(Flag.GRAPHITI_MEMORY);
  if (isOrgSettingsEnabled) enabledFlags.add(Flag.SHOW_ORG_SETTINGS);

  const items = settingsNavItems
    .filter((item) => item.flag === undefined || enabledFlags.has(item.flag))
    .map((item) => ({
      ...item,
      isActive:
        pathname === item.href ||
        (item.href !== "/settings" && pathname.startsWith(`${item.href}/`)),
    }));

  return { items };
}
