import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { AutoGPTLogo } from "./AutoGPTLogo";

const meta = {
  title: "Atoms/AutoGPTLogo",
  component: AutoGPTLogo,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "The full-colour AutoGPT logo for light surfaces. `hideText` drops the wordmark, `wordmarkColor` recolours it, and `className` replaces the default `h-10 w-22` sizing. Each instance scopes its gradient ids with `useId`, so several logos can share a page.",
      },
    },
  },
  argTypes: {
    hideText: {
      control: "boolean",
      description: "Hide the wordmark and keep only the mark",
    },
    wordmarkColor: {
      control: "color",
      description: "Fill of the wordmark path",
    },
    className: {
      control: "text",
      description: "Replaces the default size classes",
    },
  },
  args: {
    hideText: false,
  },
} satisfies Meta<typeof AutoGPTLogo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithoutWordmark: Story = {
  args: { hideText: true },
};

export const CustomWordmarkColor: Story = {
  args: {
    wordmarkColor: "currentColor",
    className: "h-10 w-auto text-zinc-600",
  },
};

export const Sizes: Story = {
  render: renderSizes,
};

export const SideBySide: Story = {
  render: renderSideBySide,
};

function renderSizes() {
  return (
    <div className="flex items-end gap-8">
      <div className="flex flex-col items-center gap-2">
        <AutoGPTLogo className="h-6 w-auto" />
        <Text variant="small" tone="muted">
          h-6
        </Text>
      </div>
      <div className="flex flex-col items-center gap-2">
        <AutoGPTLogo />
        <Text variant="small" tone="muted">
          Default (h-10)
        </Text>
      </div>
      <div className="flex flex-col items-center gap-2">
        <AutoGPTLogo className="h-16 w-auto" />
        <Text variant="small" tone="muted">
          h-16
        </Text>
      </div>
    </div>
  );
}

function renderSideBySide() {
  return (
    <div className="flex items-center gap-8">
      <AutoGPTLogo />
      <AutoGPTLogo hideText />
      <AutoGPTLogo
        wordmarkColor="currentColor"
        className="h-10 w-auto text-purple-700"
      />
    </div>
  );
}
