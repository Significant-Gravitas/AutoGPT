import type { Meta, StoryObj } from "@storybook/nextjs";
import { Kbd } from "./Kbd";

const meta: Meta<typeof Kbd> = {
  title: "Atoms/Kbd",
  component: Kbd,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          "A keyboard key chip for shortcut hints. Renders a `<kbd>`; put one key in each chip and group them next to a label.",
      },
    },
  },
  argTypes: {
    size: { control: "inline-radio", options: ["sm", "md"] },
    children: { control: "text" },
  },
  args: {
    children: "esc",
    size: "sm",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Medium: Story = {
  args: { size: "md", children: "⌘" },
};

export const Symbols: Story = {
  render: function SymbolsStory() {
    return (
      <div className="flex items-center gap-1.5">
        <Kbd>↑</Kbd>
        <Kbd>↓</Kbd>
        <Kbd>↵</Kbd>
        <Kbd>⌘</Kbd>
        <Kbd>esc</Kbd>
      </div>
    );
  },
};

export const ShortcutHints: Story = {
  render: function ShortcutHintsStory() {
    return (
      <div className="flex items-center gap-4 text-xs text-zinc-700">
        <div className="flex items-center gap-1.5">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd>
          <span>Navigate</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Kbd>↵</Kbd>
          <span>Select</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Kbd size="md">⌘</Kbd>
          <Kbd size="md">K</Kbd>
          <span>Search</span>
        </div>
      </div>
    );
  },
};
