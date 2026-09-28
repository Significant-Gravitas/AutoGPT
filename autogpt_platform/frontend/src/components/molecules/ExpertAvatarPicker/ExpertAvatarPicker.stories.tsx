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
            avatar_url: "/autogpt-characters/v2.1/expert-sofia/neutral/512.png",
          });
        }),
      ],
    },
  },
} satisfies Meta<typeof ExpertAvatarPicker>;

export default meta;
type Story = StoryObj<typeof meta>;

/** A new custom Expert: the General fallback, an upload, or a generated
 *  candidate in the chosen category's color. */
export const NewExpert: Story = {};

/** A hired built-in keeps its saved identity on offer beside the fallback. */
export const SavedIdentity: Story = {
  args: {
    name: "Maria",
    avatarUrl: "/autogpt-characters/v1.1/expert-maria/neutral/128.webp",
    categories: ["marketing", "content"],
    color: "rose-300",
  },
};
