import { Icon } from "@phosphor-icons/react";
import {
  CloudArrowUp,
  GearIcon,
  SignOut,
  SquaresFourIcon,
  UserSquareIcon,
} from "@phosphor-icons/react/dist/ssr";

type Link = {
  name: string;
  href: string;
};

export const loggedInLinks: Link[] = [
  {
    name: "Home",
    href: "/library",
  },
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

type MenuItemGroup = {
  items: {
    icon: Icon;
    text: string;
    href?: string;
    onClick?: () => void;
  }[];
};

export const accountMeunItems: MenuItemGroup[] = [
  {
    items: [
      {
        icon: UserSquareIcon,
        text: "Edit profile",
        href: "/profile",
      },
    ],
  },
  {
    items: [
      {
        icon: SquaresFourIcon,
        text: "Creator Dashboard",
        href: "/profile/dashboard",
      },
      {
        icon: CloudArrowUp,
        text: "Publish an agent",
      },
    ],
  },
  {
    items: [
      {
        icon: GearIcon,
        text: "Settings",
        href: "/profile/settings",
      },
    ],
  },
  {
    items: [
      {
        icon: SignOut,
        text: "Log out",
      },
    ],
  },
];
