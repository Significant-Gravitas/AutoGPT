import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { SwapFade } from "./SwapFade";

const meta: Meta<typeof SwapFade> = {
  title: "Atoms/SwapFade",
  component: SwapFade,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Swaps one piece of content for another in place: the outgoing content lifts and blurs away before the incoming one settles, keyed by swapKey. Honors prefers-reduced-motion with a plain crossfade.",
      },
    },
  },
  argTypes: {
    swapKey: {
      control: "text",
      description: "Key that triggers the swap when it changes.",
    },
    className: { control: "text", description: "Optional className." },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory() {
    const [label, setLabel] = useState<"Tap to talk" | "Listening...">(
      "Tap to talk",
    );

    return (
      <div className="flex flex-col items-center gap-4">
        <SwapFade swapKey={label} className="text-base text-zinc-900">
          {label}
        </SwapFade>
        <button
          type="button"
          className="rounded-full border border-zinc-300 px-3 py-1 text-sm"
          onClick={() =>
            setLabel(label === "Tap to talk" ? "Listening..." : "Tap to talk")
          }
        >
          Swap content
        </button>
      </div>
    );
  },
};
