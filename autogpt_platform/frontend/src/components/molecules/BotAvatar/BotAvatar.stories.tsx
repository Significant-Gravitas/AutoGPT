import type { Meta, StoryObj } from "@storybook/nextjs";
import { BotAvatar } from "./BotAvatar";
import {
  ACCESSORIES,
  COLORS,
  DEFAULT_CONFIG,
  SHAPES,
  STATUSES,
} from "./helpers";

const meta = {
  title: "Molecules/BotAvatar",
  component: BotAvatar,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
  args: { config: DEFAULT_CONFIG, size: 120 },
} satisfies Meta<typeof BotAvatar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const TracksPointer: Story = {
  args: { trackPointer: true, size: 160 },
};

export const AllShapes: Story = {
  render: function AllShapesStory(args) {
    return (
      <div className="flex flex-wrap items-end gap-4">
        {SHAPES.map((shape) => (
          <figure key={shape.id} className="flex flex-col items-center gap-1">
            <BotAvatar {...args} config={{ ...args.config, shape: shape.id }} />
            <figcaption className="text-xs text-muted-foreground">
              {shape.label}
            </figcaption>
          </figure>
        ))}
      </div>
    );
  },
};

export const AllColors: Story = {
  render: function AllColorsStory(args) {
    return (
      <div className="flex flex-wrap items-end gap-4">
        {COLORS.map((color) => (
          <figure key={color.id} className="flex flex-col items-center gap-1">
            <BotAvatar
              {...args}
              size={80}
              config={{ ...args.config, color: color.id }}
            />
            <figcaption className="text-xs text-muted-foreground">
              {color.label}
            </figcaption>
          </figure>
        ))}
      </div>
    );
  },
};

export const AllAccessories: Story = {
  render: function AllAccessoriesStory(args) {
    return (
      <div className="grid grid-cols-6 gap-4">
        {ACCESSORIES.map((accessory) => (
          <figure
            key={accessory.id}
            className="flex flex-col items-center gap-1"
          >
            <div className="flex items-end gap-2">
              <BotAvatar
                {...args}
                size={96}
                showBadge={false}
                config={{ ...args.config, accessory: accessory.id }}
              />
              <BotAvatar
                {...args}
                size={24}
                animated={false}
                showBadge={false}
                config={{ ...args.config, accessory: accessory.id }}
              />
            </div>
            <figcaption className="text-center text-xs text-muted-foreground">
              {accessory.label}
            </figcaption>
          </figure>
        ))}
      </div>
    );
  },
};

export const AllStatuses: Story = {
  render: function AllStatusesStory(args) {
    return (
      <div className="flex flex-wrap items-end gap-4">
        {STATUSES.map((status) => (
          <figure key={status.id} className="flex flex-col items-center gap-1">
            <BotAvatar {...args} size={88} status={status.id} />
            <figcaption className="text-xs text-muted-foreground">
              {status.label}
            </figcaption>
          </figure>
        ))}
      </div>
    );
  },
};
