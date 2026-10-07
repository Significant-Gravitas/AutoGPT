import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { AutoGPTLogoWhite } from "./AutoGPTLogoWhite";

const meta = {
  title: "Atoms/AutoGPTLogoWhite",
  component: AutoGPTLogoWhite,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="rounded-lg bg-zinc-900 p-8">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "The all-white AutoGPT logo for dark surfaces. `hideText` crops the viewBox to the mark only, and `className` replaces the default `h-13.5 w-auto` sizing.",
      },
    },
  },
  argTypes: {
    hideText: {
      control: "boolean",
      description: "Crop to the mark and drop the wordmark",
    },
    className: {
      control: "text",
      description: "Replaces the default size classes",
    },
  },
  args: {
    hideText: false,
  },
} satisfies Meta<typeof AutoGPTLogoWhite>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithoutWordmark: Story = {
  args: { hideText: true },
};

export const Sizes: Story = {
  render: renderSizes,
};

function renderSizes() {
  return (
    <div className="flex items-end gap-8">
      <div className="flex flex-col items-center gap-2">
        <AutoGPTLogoWhite className="h-6 w-auto" />
        <Text variant="small" className="text-white">
          h-6
        </Text>
      </div>
      <div className="flex flex-col items-center gap-2">
        <AutoGPTLogoWhite className="h-10 w-auto" />
        <Text variant="small" className="text-white">
          h-10
        </Text>
      </div>
      <div className="flex flex-col items-center gap-2">
        <AutoGPTLogoWhite />
        <Text variant="small" className="text-white">
          Default (h-[3.375rem])
        </Text>
      </div>
    </div>
  );
}
