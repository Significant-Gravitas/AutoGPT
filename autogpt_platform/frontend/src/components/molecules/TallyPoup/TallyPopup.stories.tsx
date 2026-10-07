import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { expect, waitFor } from "storybook/test";
import { Text } from "@/components/atoms/Text/Text";
import { TallyPopupSimple } from "./TallyPopup";

const TALLY_SCRIPT = 'script[src="https://tally.so/widgets/embed.js"]';

const meta = {
  title: "Molecules/TallyPopup",
  component: TallyPopupSimple,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "Renders nothing. On mount it appends the Tally embed script (`https://tally.so/widgets/embed.js`) to `<head>` and listens for `Tally.*` window messages; it removes both on unmount. The feedback popup itself is opened and drawn by Tally's script, so there is no visual state to show here.",
      },
    },
  },
} satisfies Meta<typeof TallyPopupSimple>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: renderWithNote,
  play: async () => {
    await waitFor(() =>
      expect(document.head.querySelector(TALLY_SCRIPT)).not.toBeNull(),
    );
  },
};

function renderWithNote() {
  return (
    <div className="flex max-w-sm flex-col gap-2">
      <TallyPopupSimple />
      <Text variant="body-medium">TallyPopup renders no markup.</Text>
      <Text variant="small" className="text-zinc-600">
        The only observable effect is the Tally embed script added to the
        document head while this story is mounted.
      </Text>
    </div>
  );
}
