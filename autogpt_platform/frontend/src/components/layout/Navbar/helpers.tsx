import {
  Book02Icon,
  CloudUploadIcon,
  CreditCardIcon,
  Logout03Icon,
  MessageMultiple02Icon,
  NewsIcon,
  PencilEdit02Icon,
  QuestionIcon,
  RefreshIcon,
  Settings01Icon,
  SlidersHorizontalIcon,
  Store01Icon,
  ToyBrickIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export enum IconType {
  Marketplace,
  Library,
  Builder,
  Edit,
  LayoutDashboard,
  UploadCloud,
  Settings,
  LogOut,
  AutoGPTLogo,
  Sliders,
  Chat,
  Billing,
  Help,
  WhatsNew,
}

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
    external?: boolean;
    onClick?: () => void;
  }[];
};

export const accountMenuItems: MenuItemGroup[] = [
  {
    items: [
      {
        icon: IconType.Edit,
        text: "Account",
        href: "/settings/account",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LayoutDashboard,
        text: "Creator Dashboard",
        href: "/settings/creator-dashboard",
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
        href: "/settings",
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

export function getAccountMenuItems(
  userRole?: string,
  newLayout = false,
): MenuItemGroup[] {
  return newLayout
    ? getNewLayoutAccountMenuItems(userRole)
    : getClassicAccountMenuItems(userRole);
}

// New sidebar layout grouping — gated behind the AUTOGPT_NEW_LAYOUT flag.
function getNewLayoutAccountMenuItems(userRole?: string): MenuItemGroup[] {
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

// Classic Navbar grouping (unchanged, pre-new-layout).
function getClassicAccountMenuItems(userRole?: string): MenuItemGroup[] {
  const baseMenuItems: MenuItemGroup[] = [
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
        {
          icon: IconType.LayoutDashboard,
          text: "Creator Dashboard",
          href: "/settings/creator-dashboard",
        },
        {
          icon: IconType.Help,
          text: "Help & Docs",
          href: "https://agpt.co/docs",
          external: true,
        },
      ],
    },
  ];

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

  baseMenuItems.push({
    items: [
      {
        icon: IconType.LogOut,
        text: "Log out",
      },
    ],
  });

  return baseMenuItems;
}

export function getAccountMenuOptionIcon(icon: IconType) {
  const iconClass = "size-4";
  switch (icon) {
    case IconType.LayoutDashboard:
      return <Icon icon={Store01Icon} className={iconClass} />;
    case IconType.UploadCloud:
      return <Icon icon={CloudUploadIcon} className={iconClass} />;
    case IconType.Edit:
      return <Icon icon={PencilEdit02Icon} className={iconClass} />;
    case IconType.Settings:
      return <Icon icon={Settings01Icon} className={iconClass} />;
    case IconType.LogOut:
      return <Icon icon={Logout03Icon} className={iconClass} />;
    case IconType.Marketplace:
      return <Icon icon={Store01Icon} className={iconClass} />;
    case IconType.Library:
      return <Icon icon={Book02Icon} className={iconClass} />;
    case IconType.Builder:
      return <Icon icon={ToyBrickIcon} className={iconClass} />;
    case IconType.Sliders:
      return <Icon icon={SlidersHorizontalIcon} className={iconClass} />;
    case IconType.Chat:
      return <Icon icon={MessageMultiple02Icon} className={iconClass} />;
    case IconType.Billing:
      return <Icon icon={CreditCardIcon} className={iconClass} />;
    case IconType.Help:
      return <Icon icon={QuestionIcon} className={iconClass} />;
    case IconType.WhatsNew:
      return <Icon icon={NewsIcon} className={iconClass} />;
    default:
      return <Icon icon={RefreshIcon} className={iconClass} />;
  }
}
