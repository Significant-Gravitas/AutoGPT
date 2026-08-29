import { Text } from "@/components/atoms/Text/Text";
import type { Meta } from "@storybook/nextjs";
import { SquareArrowOutUpRight } from "lucide-react";
import { StoryCode } from "./helpers/StoryCode";
import {
  Alert01Icon,
  Alien01Icon,
  ArrowLeft02Icon,
  ArrowRight02Icon,
  BellIcon,
  Cancel01Icon,
  CloudUploadIcon,
  Copy01Icon,
  CubeIcon,
  Delete02Icon,
  Download04Icon,
  Facebook02Icon,
  FloppyDiskIcon,
  FlowIcon,
  GithubIcon,
  GridViewIcon,
  InformationCircleIcon,
  InstagramIcon,
  Key01Icon,
  LibraryIcon,
  Linkedin01Icon,
  Logout03Icon,
  Menu01Icon,
  NewTwitterIcon,
  Package01Icon,
  PencilIcon,
  PlayIcon,
  Refresh01Icon,
  RefreshIcon,
  Settings01Icon,
  Tick02Icon,
  UserAdd01Icon,
  UserCircleIcon,
  UserIcon,
  YoutubeIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { IconSvgElement } from "@hugeicons/react";

const meta: Meta = {
  title: "Tokens /Icons",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

interface CatalogEntry {
  icon: IconSvgElement;
  /** Role the icon plays in the product. */
  name: string;
  /** Export name to import from ``@hugeicons/core-free-icons``. */
  hugeiconsName: string;
}

interface CatalogCategory {
  name: string;
  description: string;
  icons: CatalogEntry[];
}

const iconCategories: CatalogCategory[] = [
  {
    name: "User & Authentication",
    description: "Icons for user-related actions and authentication flows",
    icons: [
      { icon: UserIcon, name: "User", hugeiconsName: "UserIcon" },
      { icon: UserAdd01Icon, name: "UserPlus", hugeiconsName: "UserAdd01Icon" },
      {
        icon: UserCircleIcon,
        name: "UserCircle",
        hugeiconsName: "UserCircleIcon",
      },
      { icon: Key01Icon, name: "Key", hugeiconsName: "Key01Icon" },
      { icon: Logout03Icon, name: "SignOut", hugeiconsName: "Logout03Icon" },
    ],
  },
  {
    name: "Actions & Controls",
    description: "Icons for common user actions and interface controls",
    icons: [
      { icon: PlayIcon, name: "Play", hugeiconsName: "PlayIcon" },
      { icon: Refresh01Icon, name: "Refresh", hugeiconsName: "Refresh01Icon" },
      { icon: FloppyDiskIcon, name: "Save", hugeiconsName: "FloppyDiskIcon" },
      { icon: RefreshIcon, name: "Undo", hugeiconsName: "RefreshIcon" },
      { icon: Refresh01Icon, name: "Redo", hugeiconsName: "Refresh01Icon" },
      { icon: PencilIcon, name: "Edit", hugeiconsName: "PencilIcon" },
      { icon: Copy01Icon, name: "Copy", hugeiconsName: "Copy01Icon" },
      { icon: Delete02Icon, name: "Delete", hugeiconsName: "Delete02Icon" },
    ],
  },
  {
    name: "Navigation & Layout",
    description: "Icons for navigation, layout, and organizational elements",
    icons: [
      { icon: Menu01Icon, name: "Menu", hugeiconsName: "Menu01Icon" },
      { icon: GridViewIcon, name: "Dashboard", hugeiconsName: "GridViewIcon" },
      {
        icon: ArrowLeft02Icon,
        name: "ArrowLeft",
        hugeiconsName: "ArrowLeft02Icon",
      },
      {
        icon: ArrowRight02Icon,
        name: "ArrowRight",
        hugeiconsName: "ArrowRight02Icon",
      },
      {
        icon: Settings01Icon,
        name: "Settings",
        hugeiconsName: "Settings01Icon",
      },
      { icon: LibraryIcon, name: "Library", hugeiconsName: "LibraryIcon" },
    ],
  },
  {
    name: "Content & Media",
    description: "Icons for content types, media, and file operations",
    icons: [
      {
        icon: CloudUploadIcon,
        name: "Upload",
        hugeiconsName: "CloudUploadIcon",
      },
      {
        icon: Download04Icon,
        name: "Download",
        hugeiconsName: "Download04Icon",
      },
      { icon: Package01Icon, name: "Package", hugeiconsName: "Package01Icon" },
      { icon: CubeIcon, name: "Block", hugeiconsName: "CubeIcon" },
      { icon: FlowIcon, name: "Workflow", hugeiconsName: "FlowIcon" },
    ],
  },
  {
    name: "Feedback & Status",
    description: "Icons for alerts, notifications, and status indicators",
    icons: [
      { icon: Alert01Icon, name: "Warning", hugeiconsName: "Alert01Icon" },
      {
        icon: InformationCircleIcon,
        name: "Info",
        hugeiconsName: "InformationCircleIcon",
      },
      { icon: Tick02Icon, name: "Success", hugeiconsName: "Tick02Icon" },
      { icon: Cancel01Icon, name: "Close", hugeiconsName: "Cancel01Icon" },
      { icon: BellIcon, name: "Notification", hugeiconsName: "BellIcon" },
    ],
  },
  {
    name: "Social & External",
    description: "Icons for social media platforms and external links",
    icons: [
      { icon: GithubIcon, name: "GitHub", hugeiconsName: "GithubIcon" },
      {
        icon: Linkedin01Icon,
        name: "LinkedIn",
        hugeiconsName: "Linkedin01Icon",
      },
      {
        icon: NewTwitterIcon,
        name: "X (Twitter)",
        hugeiconsName: "NewTwitterIcon",
      },
      {
        icon: Facebook02Icon,
        name: "Facebook",
        hugeiconsName: "Facebook02Icon",
      },
      {
        icon: InstagramIcon,
        name: "Instagram",
        hugeiconsName: "InstagramIcon",
      },
      { icon: YoutubeIcon, name: "YouTube", hugeiconsName: "YoutubeIcon" },
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
            Our icon system uses Hugeicons to provide a consistent, modern, and
            comprehensive set of icons across all components. Icons ship as data
            and are rendered through the Icon atom, which applies the
            design-system stroke width for a cohesive visual language.
          </Text>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              Hugeicons
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <a
                  href="https://hugeicons.com/icons/stroke-rounded"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mb-2 inline-flex flex-row items-center gap-1 text-base font-semibold text-blue-600 hover:underline"
                >
                  Hugeicons Library{" "}
                  <SquareArrowOutUpRight className="inline-block h-3 w-3" />
                </a>
                <Text variant="body" className="mb-2 text-zinc-600">
                  A comprehensive icon family; we use the stroke-rounded variant
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  @hugeicons/core-free-icons → icon data
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-4">
                <Text
                  variant="body-medium"
                  className="mb-2 font-semibold text-zinc-800"
                >
                  Stroke Width
                </Text>
                <Text variant="body" className="mb-2 text-zinc-600">
                  There is a single stroke width, owned by the Icon atom - never
                  override it per call site
                </Text>
                <div className="space-y-1 font-mono text-sm text-zinc-800">
                  <div>2px</div>
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
                  ✅ Always Use Hugeicons
                </Text>
                <div className="space-y-2 text-blue-700">
                  <Text variant="body">
                    • Import icon data from @hugeicons/core-free-icons
                  </Text>
                  <Text variant="body">
                    • Render it with the Icon atom, never HugeiconsIcon directly
                  </Text>
                  <Text variant="body">
                    • Always match the size from Figma designs
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
              • Pick the icon whose semantics match the design, not just its
              shape
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
            <Icon icon={Alien01Icon} size={16} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              16px
            </Text>
          </div>
          <div className="flex items-center gap-4">
            <Icon icon={Alien01Icon} size={20} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              20px
            </Text>
          </div>
          <div className="flex items-center gap-4">
            <Icon icon={Alien01Icon} size={24} className="text-zinc-600" />
            <Text variant="small" className="font-mono text-zinc-500">
              24px
            </Text>
          </div>
          <div className="flex items-center gap-4">
            <Icon icon={Alien01Icon} size={32} className="text-zinc-600" />
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
              {category.icons.map((entry) => (
                <div
                  key={entry.name}
                  className="flex flex-col items-center space-y-2 rounded-lg p-3 hover:bg-gray-50"
                >
                  <Icon icon={entry.icon} size={24} className="text-zinc-600" />
                  <Text
                    variant="small"
                    className="text-center font-mono text-zinc-500"
                  >
                    {entry.hugeiconsName}
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
            How to properly implement Hugeicons in your React components.
          </Text>
        </div>

        <StoryCode
          code={`// Import icon data from Hugeicons and the Icon atom
import { UserIcon, FavouriteIcon, StarIcon, BellIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

// Basic usage - size defaults to 1em, so it scales with the text
<Icon icon={UserIcon} />
<Icon icon={FavouriteIcon} />

// Custom sizes
<Icon icon={UserIcon} size={16} />  // Small
<Icon icon={UserIcon} size={20} />  // Default
<Icon icon={UserIcon} size={24} />  // Large
<Icon icon={UserIcon} size={32} />  // Extra large

// With custom colors
<Icon icon={FavouriteIcon} className="text-red-500" />
<Icon icon={StarIcon} className="text-yellow-500" />

// Interactive states
<Icon
  icon={BellIcon}
  size={20}
  className={hasNotifications ? "text-blue-500" : "text-gray-400"}
/>

// In buttons
<button className="flex items-center gap-2">
  <Icon icon={UserIcon} size={16} />
  Profile
</button>

// Responsive sizing with Tailwind
<Icon icon={UserIcon} className="size-4 md:size-5 lg:size-6" />

// To type a prop or config entry that carries an icon
import type { IconSvgElement } from "@hugeicons/react";

interface Props {
  icon: IconSvgElement;
}`}
        />
      </div>
    </div>
  );
}
