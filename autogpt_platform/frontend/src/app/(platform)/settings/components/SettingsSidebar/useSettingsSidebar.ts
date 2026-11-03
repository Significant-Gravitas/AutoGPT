import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { usePathname } from "next/navigation";
import { settingsNavItems } from "./helpers";

export function useSettingsSidebar() {
  const pathname = usePathname();
  const isOrgSettingsEnabled = useGetFlag(Flag.SHOW_ORG_SETTINGS);
  const isBillingEnabled = useIsBillingEnabled();

  const items = settingsNavItems
    .filter((item) => item.href !== "/settings/billing" || isBillingEnabled)
    .filter(
      (item) => item.flag !== Flag.SHOW_ORG_SETTINGS || isOrgSettingsEnabled,
    )
    .map((item) => ({
      ...item,
      isActive:
        pathname === item.href ||
        (item.href !== "/settings" && pathname.startsWith(`${item.href}/`)),
    }));

  return { items };
}
