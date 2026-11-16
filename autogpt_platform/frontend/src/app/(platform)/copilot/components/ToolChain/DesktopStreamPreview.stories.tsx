import type { Meta, StoryObj } from "@storybook/nextjs";
import { DesktopStreamPreview } from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";

const meta: Meta<typeof DesktopStreamPreview> = {
  title: "Copilot/DesktopStreamPreview",
  component: DesktopStreamPreview,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "The live desktop an expert works on, as a chat tool output shows it and as the expert's Computer tab shows it (textSize body-md).",
      },
    },
  },
  args: {
    value: {
      kind: "desktop_stream",
      url: "/favicon.ico",
      provider: "e2b",
      sandbox_id: "sbx-story",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const OwnerOnly: Story = {
  args: {
    value: {
      kind: "desktop_stream",
      url: "/favicon.ico",
      provider: "e2b",
      sandbox_id: "sbx-story",
      requires_auth: true,
    },
    ownerView: false,
  },
};

export const ProfileTab: Story = {
  args: { textSize: "body-md" },
};

export const ProfileTabOwnerOnly: Story = {
  args: { ...OwnerOnly.args, textSize: "body-md" },
};
