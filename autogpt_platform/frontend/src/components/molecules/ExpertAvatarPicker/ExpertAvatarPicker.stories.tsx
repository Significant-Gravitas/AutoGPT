import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { fn } from "storybook/test";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";

const meta = {
  title: "Molecules/ExpertAvatarPicker",
  component: ExpertAvatarPicker,
  args: { name: "Nova", color: null, onPick: fn() },
  decorators: [
    (Story) => (
      <div className="w-full max-w-lg">
        <Story />
      </div>
    ),
  ],
  parameters: {
    msw: {
      handlers: [
        http.post("*/api/experts/avatars/generations", () =>
          HttpResponse.json(
            { id: "preview", status: "pending" },
            { status: 202 },
          ),
        ),
        http.get("*/api/experts/avatars/generations/preview", async () => {
          await delay(3000);
          return HttpResponse.json({
            id: "preview",
            status: "complete",
            avatar_url: "/experts/clay/v1/finance.png",
          });
        }),
      ],
    },
  },
} satisfies Meta<typeof ExpertAvatarPicker>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Catalog: Story = {};
export const ExistingAvatar: Story = {
  args: { avatarUrl: "/experts/clay/v1/marketing.png", color: "rose-300" },
};
