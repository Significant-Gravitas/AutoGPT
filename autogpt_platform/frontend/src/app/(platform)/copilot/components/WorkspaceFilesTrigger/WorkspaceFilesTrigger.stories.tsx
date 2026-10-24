import type { Meta, StoryObj } from "@storybook/nextjs";
import { WorkspaceFilesTrigger } from "./WorkspaceFilesTrigger";

const meta: Meta<typeof WorkspaceFilesTrigger> = {
  title: "Copilot/WorkspaceFilesTrigger",
  component: WorkspaceFilesTrigger,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Folder button in the chat header that toggles the files tab of the side panel.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
