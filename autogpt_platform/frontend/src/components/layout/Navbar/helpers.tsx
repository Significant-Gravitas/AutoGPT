import {
  IconBuilder,
  IconEdit,
  IconLibrary,
  IconLogOut,
  IconMarketplace,
  IconRefresh,
  IconSettings,
  IconSliders,
  IconType,
  IconUploadCloud,
} from "@/components/__legacy__/ui/icons";
import { ChatsIcon, StorefrontIcon } from "@phosphor-icons/react";

type Link = {
  name: string;
  href: string;
};

export const loggedInLinks: Link[] = [
  {
    name: "Marketplace",
    href: "/marketplace",
  },
  {
    name: "Library",
    href: "/library",
  },
  {
    name: "Build",
    href: "/build",
  },
];

export const loggedOutLinks: Link[] = [
  {
    name: "Marketplace",
    href: "/marketplace",
  },
];

export type MenuItemGroup = {
  groupName?: string;
  items: {
    icon: IconType;
    text: string;
    href?: string;
    onClick?: () => void;
  }[];
};

export const accountMenuItems: MenuItemGroup[] = [
  {
    items: [
      {
        icon: IconType.Edit,
        text: "Edit profile",
        href: "/profile",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LayoutDashboard,
        text: "Creator Dashboard",
        href: "/profile/dashboard",
      },
      {
        icon: IconType.UploadCloud,
        text: "Publish an agent",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.Settings,
        text: "Settings",
        href: "/profile/settings",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LogOut,
        text: "Log out",
      },
    ],
  },
];

export function getAccountMenuItems(userRole?: string): MenuItemGroup[] {
  const baseMenuItems: MenuItemGroup[] = [
    {
      items: [
        {
          icon: IconType.Edit,
          text: "Edit profile",
          href: "/profile",
        },
      ],
    },
    {
      items: [
        {
          icon: IconType.LayoutDashboard,
          text: "Creator Dashboard",
          href: "/profile/dashboard",
        },
        {
          icon: IconType.UploadCloud,
          text: "Publish an agent",
        },
      ],
    },
  ];

  // Add admin menu item for admin users
  if (userRole === "admin") {
    baseMenuItems.push({
      items: [
        {
          icon: IconType.Sliders,
          text: "Admin",
          href: "/admin/marketplace",
        },
      ],
    });
  }

  // Add settings and logout
  baseMenuItems.push(
    {
      items: [
        {
          icon: IconType.Settings,
          text: "Settings",
          href: "/profile/settings",
        },
      ],
    },
    {
      items: [
        {
          icon: IconType.LogOut,
          text: "Log out",
        },
      ],
    },
  );

  return baseMenuItems;
}

export function getAccountMenuOptionIcon(icon: IconType) {
  const iconClass = "w-5 h-5";
  switch (icon) {
    case IconType.LayoutDashboard:
      return <StorefrontIcon className={iconClass} />;
    case IconType.UploadCloud:
      return <IconUploadCloud className={iconClass} />;
    case IconType.Edit:
      return <IconEdit className={iconClass} />;
    case IconType.Settings:
      return <IconSettings className={iconClass} />;
    case IconType.LogOut:
      return <IconLogOut className={iconClass} />;
    case IconType.Marketplace:
      return <IconMarketplace className={iconClass} />;
    case IconType.Library:
      return <IconLibrary className={iconClass} />;
    case IconType.Builder:
      return <IconBuilder className={iconClass} />;
    case IconType.Sliders:
      return <IconSliders className={iconClass} />;
    case IconType.Chat:
      return <ChatsIcon className={iconClass} />;
    default:
      return <IconRefresh className={iconClass} />;
  }
}
