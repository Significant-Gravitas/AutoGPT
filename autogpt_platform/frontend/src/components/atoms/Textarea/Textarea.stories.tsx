import type { Meta, StoryObj } from "@storybook/nextjs";
import { Textarea } from "./Textarea";

const meta: Meta<typeof Textarea> = {
  title: "Atoms/Textarea",
  component: Textarea,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          'Bare multi-line text field, without a label or error slot. For a labelled field use the Input atom with type="textarea".',
      },
    },
  },
  argTypes: {
    disabled: { control: "boolean", description: "Disable the field." },
    placeholder: { control: "text", description: "Placeholder text." },
    rows: { control: "number", description: "Visible rows." },
    className: { control: "text", description: "Optional className." },
  },
  args: { "aria-label": "Notes", placeholder: "Add any notes" },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {};

export const Filled: Story = {
  args: { defaultValue: "Meets all requirements" },
};

export const Disabled: Story = {
  args: { disabled: true, defaultValue: "Meets all requirements" },
};
