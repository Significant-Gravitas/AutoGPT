import {
  IconBuilder,
  IconEdit,
  IconLayoutDashboard,
  IconLibrary,
  IconLogOut,
  IconMarketplace,
  IconRefresh,
  IconSettings,
  IconType,
  IconUploadCloud,
} from "@/components/ui/icons";

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

export function getAccountMenuOptionIcon(icon: IconType) {
  const iconClass = "w-6 h-6";
  switch (icon) {
    case IconType.LayoutDashboard:
      return <IconLayoutDashboard className={iconClass} />;
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
    default:
      return <IconRefresh className={iconClass} />;
  }
}
