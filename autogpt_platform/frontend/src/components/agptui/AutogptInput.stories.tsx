import { Meta, StoryObj } from "@storybook/react";
import AutogptInput from "./AutogptInput";

const meta: Meta<typeof AutogptInput> = {
  title: "new/AutogptInput",
  component: AutogptInput,
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center bg-[#E1E1E1] p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof AutogptInput>;

export const Default: Story = {
  args: {
    label: "Email",
    placeholder: "Type something...",
    onChange: () => {
      console.log("It's working");
    },
  },
};

export const IsDisabled: Story = {
  args: {
    label: "Email",
    placeholder: "Type something...",
    isDisabled: true,
    onChange: () => {
      console.log("It's working");
    },
  },
};
