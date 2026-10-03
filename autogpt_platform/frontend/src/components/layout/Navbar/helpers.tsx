import { IconType } from "@/components/__legacy__/ui/icons";

export type MenuItemGroup = {
  groupName?: string;
  items: {
    icon: IconType;
    text: string;
    href?: string;
    external?: boolean;
    onClick?: () => void;
  }[];
};

export function getAccountMenuItems(userRole?: string): MenuItemGroup[] {
  const footerItems: MenuItemGroup["items"] = [
    {
      icon: IconType.WhatsNew,
      text: "What's new",
      href: "https://agpt.co/docs/platform/changelog/changelog/",
      external: true,
    },
    {
      icon: IconType.Help,
      text: "Help & Docs",
      href: "https://agpt.co/docs",
      external: true,
    },
  ];

  if (userRole === "admin") {
    footerItems.push({
      icon: IconType.Sliders,
      text: "Admin",
      href: "/admin/marketplace",
    });
  }

  footerItems.push({
    icon: IconType.LogOut,
    text: "Log out",
  });

  return [
    {
      items: [
        {
          icon: IconType.Edit,
          text: "Profile",
          href: "/settings/profile",
        },
        {
          icon: IconType.Settings,
          text: "Settings",
          href: "/settings/account",
        },
        {
          icon: IconType.Billing,
          text: "Billing",
          href: "/settings/billing",
        },
      ],
    },
    {
      items: footerItems,
    },
  ];
}
