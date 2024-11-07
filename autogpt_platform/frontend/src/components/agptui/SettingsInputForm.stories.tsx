import type { Meta, StoryObj } from "@storybook/react";
import { SettingsInputForm } from "./SettingsInputForm";

const meta: Meta<typeof SettingsInputForm> = {
  title: "AGPT UI/Settings/Settings Input Form",
  component: SettingsInputForm,
  parameters: {
    layout: "fullscreen",
  },
};

export default meta;
type Story = StoryObj<typeof SettingsInputForm>;

export const Default: Story = {
  args: {
    email: "johndoe@email.com",
    desktopNotifications: {
      first: false,
      second: true,
    },
  },
};
