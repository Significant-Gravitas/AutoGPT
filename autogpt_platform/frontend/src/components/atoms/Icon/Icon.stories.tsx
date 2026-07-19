import type { Meta, StoryObj } from "@storybook/nextjs";
import { Icon } from "./Icon";
import { iconRegistry, type IconName } from "./registry";

const meta: Meta<typeof Icon> = {
  title: "Atoms/Icon",
  tags: ["autodocs"],
  component: Icon,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Semantic icon that renders from the AutoGPT icon set when the optional `@autogpt/icons` package is installed, and falls back to the matching Phosphor icon otherwise. Add new icons to `registry.ts`.",
      },
    },
  },
  argTypes: {
    name: {
      control: "select",
      options: Object.keys(iconRegistry),
      description: "Semantic icon name from the registry",
    },
    size: { control: "number", description: "Icon size in pixels" },
    color: { control: "color", description: "Icon color" },
  },
};

export default meta;
type Story = StoryObj<typeof Icon>;

export const Default: Story = {
  args: { name: "home", size: 32 },
};

export const AllIcons: Story = {
  render: () => (
    <div className="grid grid-cols-4 gap-6">
      {(Object.keys(iconRegistry) as IconName[]).map((name) => (
        <div key={name} className="flex flex-col items-center gap-2">
          <Icon name={name} size={28} />
          <span className="text-xs text-zinc-500">{name}</span>
        </div>
      ))}
    </div>
  ),
};
