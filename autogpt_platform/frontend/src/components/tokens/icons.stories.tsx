import { Text } from "@/components/atoms/Text/Text";
import {
  Alien,
  ArrowClockwise,
  ArrowCounterClockwise,
  ArrowLeft,
  ArrowRight,
  Bell,
  Books,
  Check,
  CloudArrowUp,
  Copy,
  Cube,
  Download,
  FacebookLogo,
  FloppyDisk,
  FlowArrow,
  Gear,
  GithubLogo,
  Info,
  InstagramLogo,
  Key,
  LinkedinLogo,
  List,
  Package,
  PencilSimple,
  Play,
  ArrowClockwise as Redo,
  SignOut,
  SquaresFour,
  Trash,
  User,
  UserCircle,
  UserPlus,
  Warning,
  X,
  XLogo,
  YoutubeLogo,
} from "@phosphor-icons/react";
import type { Meta } from "@storybook/nextjs";
import { SquareArrowOutUpRight } from "lucide-react";
import { StoryCode } from "./helpers/StoryCode";

const meta: Meta = {
  title: "Tokens /Icons",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

// Icon categories with examples
const iconCategories = [
  {
    name: "User & Authentication",
    description: "Icons for user-related actions and authentication flows",
    icons: [
      { component: User, name: "User", phosphorName: "User" },
      { component: UserPlus, name: "UserPlus", phosphorName: "UserPlus" },
      { component: UserCircle, name: "UserCircle", phosphorName: "UserCircle" },
      { component: Key, name: "Key", phosphorName: "Key" },
      { component: SignOut, name: "SignOut", phosphorName: "SignOut" },
    ],
  },
  {
    name: "Actions & Controls",
    description: "Icons for common user actions and interface controls",
    icons: [
      { component: Play, name: "Play", phosphorName: "Play" },
      {
        component: ArrowClockwise,
        name: "Refresh",
        phosphorName: "ArrowClockwise",
      },
      { component: FloppyDisk, name: "Save", phosphorName: "FloppyDisk" },
      {
        component: ArrowCounterClockwise,
        name: "Undo",
        phosphorName: "ArrowCounterClockwise",
      },
      { component: Redo, name: "Redo", phosphorName: "ArrowClockwise" },
      { component: PencilSimple, name: "Edit", phosphorName: "PencilSimple" },
      { component: Copy, name: "Copy", phosphorName: "Copy" },
      { component: Trash, name: "Delete", phosphorName: "Trash" },
    ],
  },
  {
    name: "Navigation & Layout",
    description: "Icons for navigation, layout, and organizational elements",
    icons: [
      { component: List, name: "Menu", phosphorName: "List" },
      {
        component: SquaresFour,
        name: "Dashboard",
        phosphorName: "SquaresFour",
      },
      { component: ArrowLeft, name: "ArrowLeft", phosphorName: "ArrowLeft" },
      { component: ArrowRight, name: "ArrowRight", phosphorName: "ArrowRight" },
      { component: Gear, name: "Settings", phosphorName: "Gear" },
      { component: Books, name: "Library", phosphorName: "Books" },
    ],
  },
  {
    name: "Content & Media",
    description: "Icons for content types, media, and file operations",
    icons: [
      { component: CloudArrowUp, name: "Upload", phosphorName: "CloudArrowUp" },
      { component: Download, name: "Download", phosphorName: "Download" },
      { component: Package, name: "Package", phosphorName: "Package" },
      { component: Cube, name: "Block", phosphorName: "Cube" },
      { component: FlowArrow, name: "Workflow", phosphorName: "FlowArrow" },
    ],
  },
  {
    name: "Feedback & Status",
    description: "Icons for alerts, notifications, and status indicators",
    icons: [
      { component: Warning, name: "Warning", phosphorName: "Warning" },
      { component: Info, name: "Info", phosphorName: "Info" },
      { component: Check, name: "Success", phosphorName: "Check" },
      { component: X, name: "Close", phosphorName: "X" },
      { component: Bell, name: "Notification", phosphorName: "Bell" },
    ],
  },
  {
    name: "Social & External",
    description: "Icons for social media platforms and external links",
    icons: [
      { component: GithubLogo, name: "GitHub", phosphorName: "GithubLogo" },
      {
        component: LinkedinLogo,
        name: "LinkedIn",
        phosphorName: "LinkedinLogo",
      },
      { component: XLogo, name: "X (Twitter)", phosphorName: "XLogo" },
      {
        component: FacebookLogo,
        name: "Facebook",
        phosphorName: "FacebookLogo",
      },
      {
        component: InstagramLogo,
        name: "Instagram",
        phosphorName: "InstagramLogo",
      },
      { component: YoutubeLogo, name: "YouTube", phosphorName: "YoutubeLogo" },
    ],
  },
];

export function AllVariants() {
  return (
    <div className="space-y-12">
      {/* Icons System Documentation */}
      <div className="space-y-8">
        <div>
          <Text variant="h1" className="mb-4 text-zinc-800">
            Icons System
          </Text>
          <Text variant="large" className="text-zinc-600">
            Our icon system uses Phosphor Icons to provide a consistent, modern,
            and comprehensive set of icons across all components. Phosphor
            offers multiple weights and a cohesive design language that aligns
            with our design principles.
          </Text>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              Phosphor Icons
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <a
                  href="https://phosphoricons.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mb-2 inline-flex flex-row items-center gap-1 text-base font-semibold text-blue-600 hover:underline"
                >
                  Phosphor Icons Library{" "}
                  <SquareArrowOutUpRight className="inline-block h-3 w-3" />
                </a>
                <Text variant="body" className="mb-2 text-zinc-600">
                  A flexible icon family with multiple weights and styles
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  @phosphor-icons/react → React components
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-4">
                <Text
                  variant="body-medium"
                  className="mb-2 font-semibold text-zinc-800"
                >
                  Available Weights
                </Text>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Phosphor icons offer multiple weights - use the one specified
                  in your designs
                </Text>
                <div className="space-y-1 font-mono text-sm text-zinc-800">
                  <div>regular (default), light, bold, fill, thin, duotone</div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              Usage Guidelines
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border-2 border-dashed border-blue-200 bg-blue-50 p-4">
                <Text
                  variant="body-medium"
                  className="mb-2 font-semibold text-blue-800"
                >
                  ✅ Always Use Phosphor Icons
                </Text>
                <div className="space-y-2 text-blue-700">
                  <Text variant="body">
                    • Import from @phosphor-icons/react
                  </Text>
                  <Text variant="body">
                    • Always match size and weight from Figma designs
                  </Text>
                  <Text variant="body">
                    • Ensure icons have proper semantic meaning
                  </Text>
                  <Text variant="body">
                    • Verify accessibility and color contrast
                  </Text>
                </div>
              </div>
              <div>
                <Text
                  variant="h3"
                  className="mb-2 text-base font-semibold text-zinc-800"
                >
                  🎨 Design Consistency
                </Text>
                <div className="space-y-2 text-zinc-600">
                  <Text variant="body">
                    • Follow the exact specifications from design team
                  </Text>
                  <Text variant="body">
                    • Maintain consistency across similar UI elements
                  </Text>
                  <Text variant="body">
                    • Consider accessibility requirements (minimum 16px)
                  </Text>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Design Matching */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-2 text-xl font-semibold text-zinc-800"
          >
            Matching Design Specifications
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            When implementing icons, always reference the design specifications
            provided by the design team to ensure proper sizing and weight.
          </Text>
        </div>

        <div className="rounded-lg border-2 border-dashed border-amber-200 bg-amber-50 p-6">
          <Text
            variant="body-medium"
            className="mb-3 font-semibold text-amber-800"
          >
            🎨 Always Match Figma Designs
          </Text>
          <div className="space-y-3 text-amber-700">
            <Text variant="body">
              • Check the Figma designs for exact icon sizes (16px, 20px, 24px,
              etc.)
            </Text>
            <Text variant="body">
              • Match the icon weight specified in designs (regular, bold, fill,
              etc.)
            </Text>
            <Text variant="body">
              • Ensure color and opacity match the design specifications
            </Text>
            <Text variant="body">
              • Verify spacing and alignment with surrounding elements
            </Text>
          </div>
        </div>

        <div className="flex items-center gap-8 rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-4">
            <Alien size={16} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              16px
            </Text>
          </div>
          <div className="flex items-center gap-4">
            <Alien size={20} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              20px
            </Text>
          </div>
          <div className="flex items-center gap-4">
            <Alien size={24} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              24px
            </Text>
          </div>
          <div className="flex items-center gap-4">
            <Alien size={32} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              32px
            </Text>
          </div>
        </div>
      </div>

      {/* Icon Categories */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-2 text-xl font-semibold text-zinc-800"
          >
            Icon Categories
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            Our curated icon set organized by functional categories. Each icon
            is carefully selected to maintain consistency and semantic clarity.
          </Text>
        </div>

        {iconCategories.map((category) => (
          <div key={category.name} className="space-y-4">
            <div>
              <Text
                variant="h3"
                className="mb-1 text-lg font-semibold text-zinc-800"
              >
                {category.name}
              </Text>
              <Text variant="body" className="text-zinc-600">
                {category.description}
              </Text>
            </div>
            <div className="grid grid-cols-2 gap-4 rounded-lg border border-gray-200 p-4 md:grid-cols-3 lg:grid-cols-6">
              {category.icons.map((icon) => (
                <div
                  key={icon.name}
                  className="flex flex-col items-center space-y-2 rounded-lg p-3 hover:bg-gray-50"
                >
                  <icon.component size={24} className="text-zinc-600" />
                  <Text
                    variant="small"
                    className="text-center font-mono text-zinc-500"
                  >
                    {icon.phosphorName}
                  </Text>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Usage Examples */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-2 text-xl font-semibold text-zinc-800"
          >
            Usage Examples
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            How to properly implement Phosphor icons in your React components.
          </Text>
        </div>

        <StoryCode
          code={`// Import icons from Phosphor
import { User, Heart, Star, Bell } from "@phosphor-icons/react";

// Basic usage with default size (24px)
<User />
<Heart />

// Custom sizes
<User size={16} />  // Small
<User size={20} />  // Default
<User size={24} />  // Large
<User size={32} />  // Extra large

// With custom colors
<Heart className="text-red-500" />
<Star className="text-yellow-500" />

// Different weights
<User weight="thin" />     // 1px stroke
<User weight="light" />    // 1.5px stroke  
<User weight="regular" />  // 2px stroke (default)
<User weight="bold" />     // 2.5px stroke
<User weight="fill" />     // Filled version
<User weight="duotone" />  // Two-tone style

// Interactive states
<Bell 
  size={20}
  weight={hasNotifications ? "fill" : "regular"}
  className={hasNotifications ? "text-blue-500" : "text-gray-400"}
/>

// In buttons
<button className="flex items-center gap-2">
  <User size={16} />
  Profile
</button>

// Responsive sizing with Tailwind
<User className="size-4 md:size-5 lg:size-6" />`}
        />
      </div>
    </div>
  );
}
