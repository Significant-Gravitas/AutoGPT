import type { Meta, StoryObj } from "@storybook/nextjs";
import { OverflowText } from "./OverflowText";

const meta: Meta<typeof OverflowText> = {
  title: "Atoms/OverflowText",
  component: OverflowText,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Text component that automatically truncates overflowing content with ellipsis and shows a tooltip on hover when truncated. Supports both string and ReactNode values.",
      },
    },
  },
  argTypes: {
    value: {
      control: "text",
      description: "The text content to display (string or ReactNode)",
    },
    className: {
      control: "text",
      description: "Additional CSS classes to customize styling",
    },
  },
  args: {
    value: "This is a sample text that may overflow",
    className: "",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: function DefaultOverflowText(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} />
      </div>
    );
  },
};

export const ShortText: Story = {
  args: {
    value: "Short text",
  },
  render: function ShortTextStory(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} />
      </div>
    );
  },
};

export const LongText: Story = {
  args: {
    value:
      "This is a very long text that will definitely overflow and show a tooltip when you hover over it",
  },
  render: function LongTextStory(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} />
      </div>
    );
  },
};

export const CustomStyling: Story = {
  args: {
    value: "Text with custom styling",
    className: "text-lg font-semibold text-indigo-600",
  },
  render: function CustomStylingStory(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} />
      </div>
    );
  },
};

export const WithReactNode: Story = {
  args: {
    value: (
      <span>
        Text with <strong>bold</strong> and <em>italic</em> content
      </span>
    ),
  },
  render: function WithReactNodeStory(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} />
      </div>
    );
  },
};

export const DifferentWidths: Story = {
  render: function DifferentWidthsStory() {
    const longText =
      "This text will truncate differently depending on the container width";
    return (
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <span className="text-xs text-zinc-500">Width: 200px</span>
          <div className="w-[200px]">
            <OverflowText value={longText} variant="body" />
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <span className="text-xs text-zinc-500">Width: 300px</span>
          <div className="w-[300px]">
            <OverflowText value={longText} variant="body" />
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <span className="text-xs text-zinc-500">Width: 400px</span>
          <div className="w-[400px]">
            <OverflowText value={longText} variant="body" />
          </div>
        </div>
      </div>
    );
  },
};

export const FilePathExample: Story = {
  args: {
    value: "/very/long/path/to/a/file/that/might/overflow/in/the/ui.tsx",
  },
  render: function FilePathExampleStory(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} className="font-mono text-sm" />
      </div>
    );
  },
};

export const URLExample: Story = {
  args: {
    value: "https://example.com/very/long/url/path/that/might/overflow",
  },
  render: function URLExampleStory(args) {
    return (
      <div className="w-64">
        <OverflowText {...args} className="text-blue-600" />
      </div>
    );
  },
};
