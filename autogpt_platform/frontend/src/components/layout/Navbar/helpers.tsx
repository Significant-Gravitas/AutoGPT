import { IconType } from "@/components/__legacy__/ui/icons";
import { fixedCopy, type FixedCopy } from "@/lib/replay-privacy";

export type MenuItemGroup = {
  groupName?: string;
  items: {
    icon: IconType;
    text: FixedCopy;
    href?: string;
    external?: boolean;
    onClick?: () => void;
  }[];
};

export function getAccountMenuItems(
  userRole?: string,
  isBillingEnabled = true,
): MenuItemGroup[] {
  const footerItems: MenuItemGroup["items"] = [
    {
      icon: IconType.WhatsNew,
      text: fixedCopy("What's new"),
      href: "https://agpt.co/docs/platform/changelog/changelog/",
      external: true,
    },
    {
      icon: IconType.Help,
      text: fixedCopy("Help & Docs"),
      href: "https://agpt.co/docs",
      external: true,
    },
  ];

  if (userRole === "admin") {
    footerItems.push({
      icon: IconType.Sliders,
      text: fixedCopy("Admin"),
      href: "/admin/marketplace",
    });
  }

  footerItems.push({
    icon: IconType.LogOut,
    text: fixedCopy("Log out"),
  });

  return [
    {
      items: [
        {
          icon: IconType.Edit,
          text: fixedCopy("Profile"),
          href: "/settings/profile",
        },
        {
          icon: IconType.Settings,
          text: fixedCopy("Settings"),
          href: "/settings/account",
        },
        ...(isBillingEnabled
          ? [
              {
                icon: IconType.Billing,
                text: fixedCopy("Billing"),
                href: "/settings/billing",
              },
            ]
          : []),
      ],
    },
    {
      items: footerItems,
    },
  ];
}
