import { Text } from "@/components/atoms/Text/Text";
import {
  Alert01Icon,
  ArrowLeft01Icon,
  Calendar03Icon,
  Cancel01Icon,
  Clock01Icon,
  Delete02Icon,
  FavouriteIcon,
  Home01Icon,
  InformationCircleIcon,
  Mail01Icon,
  Notification01Icon,
  PencilEdit02Icon,
  PlusSignIcon,
  Robot01Icon,
  Search01Icon,
  Settings01Icon,
  SparklesIcon,
  StarIcon,
  Tick02Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { Icon } from "./Icon";

const meta = {
  title: "Atoms/Icon",
  component: Icon,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          'The only way to render icons: a thin wrapper over `HugeiconsIcon` that takes icon data from `@hugeicons/core-free-icons`. Defaults to `size="1em"` (inherits the font size) and a stroke width of 2. Colour follows `currentColor`, so set it with a `text-*` class. Icons are decorative by default; give a meaningful standalone icon `role="img"` and an `aria-label`.',
      },
    },
  },
  argTypes: {
    size: {
      control: "number",
      description: 'Width and height in px, or any CSS length. Default "1em"',
    },
    strokeWidth: {
      control: { type: "range", min: 0.5, max: 3, step: 0.5 },
      description: "Stroke width. Default 2",
    },
    className: {
      control: "text",
      description: "Classes on the svg, e.g. text colour",
    },
  },
  args: {
    icon: SparklesIcon,
    size: 24,
  },
} satisfies Meta<typeof Icon>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const InheritsFontSize: Story = {
  args: { size: undefined },
  render: renderInheritsFontSize,
};

export const Sizes: Story = {
  render: renderSizes,
};

export const StrokeWidths: Story = {
  render: renderStrokeWidths,
};

export const Colors: Story = {
  render: renderColors,
};

export const Labelled: Story = {
  args: {
    icon: Alert01Icon,
    role: "img",
    "aria-label": "Warning",
    className: "text-red-600",
  },
};

export const Gallery: Story = {
  render: renderGallery,
};

const SIZES = [12, 16, 20, 24, 32, 48];

const STROKE_WIDTHS = [1, 1.5, 2, 2.5];

const COLORS = [
  { name: "zinc-900", className: "text-zinc-900" },
  { name: "zinc-500", className: "text-zinc-500" },
  { name: "red-600", className: "text-red-600" },
  { name: "green-600", className: "text-green-600" },
  { name: "yellow-600", className: "text-yellow-600" },
  { name: "purple-600", className: "text-purple-600" },
];

const GALLERY = [
  { name: "Home01Icon", icon: Home01Icon },
  { name: "Search01Icon", icon: Search01Icon },
  { name: "Settings01Icon", icon: Settings01Icon },
  { name: "Notification01Icon", icon: Notification01Icon },
  { name: "UserIcon", icon: UserIcon },
  { name: "Mail01Icon", icon: Mail01Icon },
  { name: "Calendar03Icon", icon: Calendar03Icon },
  { name: "Clock01Icon", icon: Clock01Icon },
  { name: "PlusSignIcon", icon: PlusSignIcon },
  { name: "PencilEdit02Icon", icon: PencilEdit02Icon },
  { name: "Delete02Icon", icon: Delete02Icon },
  { name: "Tick02Icon", icon: Tick02Icon },
  { name: "Cancel01Icon", icon: Cancel01Icon },
  { name: "ArrowLeft01Icon", icon: ArrowLeft01Icon },
  { name: "Alert01Icon", icon: Alert01Icon },
  { name: "InformationCircleIcon", icon: InformationCircleIcon },
  { name: "SparklesIcon", icon: SparklesIcon },
  { name: "Robot01Icon", icon: Robot01Icon },
  { name: "StarIcon", icon: StarIcon },
  { name: "FavouriteIcon", icon: FavouriteIcon },
];

function renderInheritsFontSize() {
  return (
    <div className="flex flex-col gap-3">
      <Text variant="small" className="flex items-center gap-2">
        <Icon icon={SparklesIcon} /> Small text
      </Text>
      <Text variant="body" className="flex items-center gap-2">
        <Icon icon={SparklesIcon} /> Body text
      </Text>
      <Text variant="h4" as="p" className="flex items-center gap-2">
        <Icon icon={SparklesIcon} /> Heading text
      </Text>
    </div>
  );
}

function renderSizes() {
  return (
    <div className="flex items-end gap-6">
      {SIZES.map((size) => (
        <div key={size} className="flex flex-col items-center gap-2">
          <Icon icon={SparklesIcon} size={size} />
          <Text variant="small" tone="muted">
            {size}px
          </Text>
        </div>
      ))}
    </div>
  );
}

function renderStrokeWidths() {
  return (
    <div className="flex items-end gap-6">
      {STROKE_WIDTHS.map((strokeWidth) => (
        <div key={strokeWidth} className="flex flex-col items-center gap-2">
          <Icon icon={Settings01Icon} size={32} strokeWidth={strokeWidth} />
          <Text variant="small" tone="muted">
            {strokeWidth}
          </Text>
        </div>
      ))}
    </div>
  );
}

function renderColors() {
  return (
    <div className="flex items-end gap-6">
      {COLORS.map((color) => (
        <div key={color.name} className="flex flex-col items-center gap-2">
          <Icon icon={FavouriteIcon} size={32} className={color.className} />
          <Text variant="small" tone="muted">
            {color.name}
          </Text>
        </div>
      ))}
    </div>
  );
}

function renderGallery() {
  return (
    <div className="grid grid-cols-5 gap-4">
      {GALLERY.map((item) => (
        <div
          key={item.name}
          className="flex w-36 flex-col items-center gap-2 rounded-lg border border-zinc-200 p-3"
        >
          <Icon icon={item.icon} size={24} className="text-zinc-800" />
          <Text variant="small" tone="secondary">
            {item.name}
          </Text>
        </div>
      ))}
    </div>
  );
}
