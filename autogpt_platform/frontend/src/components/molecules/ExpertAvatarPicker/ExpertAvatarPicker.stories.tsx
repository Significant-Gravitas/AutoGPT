import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { fn, userEvent, within } from "storybook/test";
import {
  DEFAULT_EXPERT_AVATAR_URL,
  MANAGED_IDENTITIES,
} from "../ExpertAvatar/helpers";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";

const meta = {
  title: "Molecules/ExpertAvatarPicker",
  component: ExpertAvatarPicker,
  args: { name: "Nova", category: "finance", onPick: fn() },
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
            avatar_url: DEFAULT_EXPERT_AVATAR_URL,
          });
        }),
      ],
    },
  },
} satisfies Meta<typeof ExpertAvatarPicker>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Generating: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Regenerate" }),
    );
  },
};

/** The team page: the expert already has a face until it is regenerated. */
export const ExistingAvatar: Story = {
  args: { category: "marketing", avatarUrl: MANAGED_IDENTITIES[0].url },
};
