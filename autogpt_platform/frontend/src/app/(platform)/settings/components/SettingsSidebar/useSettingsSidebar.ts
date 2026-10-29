import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { usePathname } from "next/navigation";
import { settingsNavItems } from "./helpers";

export function useSettingsSidebar() {
  const pathname = usePathname();

  const isBillingEnabled = useIsBillingEnabled();

  const items = settingsNavItems
    .filter((item) => item.href !== "/settings/billing" || isBillingEnabled)
    .map((item) => ({
      ...item,
      isActive:
        pathname === item.href ||
        (item.href !== "/settings" && pathname.startsWith(`${item.href}/`)),
    }));

  return { items };
}
