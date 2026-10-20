import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { userEvent, within } from "storybook/test";
import { ChangelogPopup } from "./ChangelogPopup";
import { STORAGE_KEY } from "./changelog-constants";

const INDEX_MD = `
| Date | Highlights |
| ---- | ---------- |
| [June 18 – July 9](/docs/platform/changelog/changelog/june-18-july-9-2026.md) | File folders, shareable chats, Discord bot |
`;

const meta = {
  title: "Molecules/ChangelogPopup",
  component: ChangelogPopup,
  parameters: {
    layout: "fullscreen",
    msw: {
      handlers: [
        http.get("*/api/changelog", () => HttpResponse.text(INDEX_MD)),
      ],
    },
  },
  beforeEach() {
    localStorage.removeItem(STORAGE_KEY);
  },
} satisfies Meta<typeof ChangelogPopup>;

export default meta;
type Story = StoryObj<typeof meta>;

export const LatestRelease: Story = {
  render: function Render() {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-background">
        <Text variant="h3">Latest release</Text>
        <Text variant="body" className="mt-2 text-muted-foreground">
          The notification appears in the bottom corner. Hover or focus to
          pause.
        </Text>
        <ChangelogPopup />
      </div>
    );
  },
  play: async ({ canvasElement }) => {
    const popup = await within(canvasElement).findByRole(
      "dialog",
      { name: "What's new" },
      { timeout: 5000 },
    );
    await userEvent.hover(popup);
  },
};
