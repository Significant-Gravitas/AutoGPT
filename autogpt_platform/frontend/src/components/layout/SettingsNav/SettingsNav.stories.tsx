import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  Coins01Icon,
  ElectricPlugsIcon,
  Key01Icon,
  SlidersHorizontalIcon,
  Store01Icon,
  UserCircleIcon,
} from "@hugeicons/core-free-icons";
import { SettingsNav } from "./SettingsNav";

const LINKS = [
  { label: "Profile", href: "/profile", icon: UserCircleIcon },
  { label: "Creator Dashboard", href: "/profile/dashboard", icon: Store01Icon },
  { label: "Billing", href: "/profile/credits", icon: Coins01Icon },
  {
    label: "Integrations",
    href: "/profile/integrations",
    icon: ElectricPlugsIcon,
  },
  { label: "Settings", href: "/profile/settings", icon: SlidersHorizontalIcon },
  { label: "API Keys", href: "/profile/api-keys", icon: Key01Icon },
];

const meta = {
  title: "Layout/SettingsNav",
  component: SettingsNav,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    a11y: { test: "error" },
    nextjs: { navigation: { pathname: "/profile" } },
    docs: {
      description: {
        component:
          "Vertical link list for the settings and admin pages. A panel on large screens; a menu button that opens a Sheet below `lg`. The link whose href matches the current pathname is marked `aria-current`.",
      },
    },
  },
  args: {
    label: "Settings",
    links: LINKS,
  },
} satisfies Meta<typeof SettingsNav>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const NestedRouteActive: Story = {
  parameters: {
    nextjs: { navigation: { pathname: "/profile/dashboard/submissions" } },
  },
};

export const Mobile: Story = {
  parameters: { viewport: { defaultViewport: "mobile1" } },
};
