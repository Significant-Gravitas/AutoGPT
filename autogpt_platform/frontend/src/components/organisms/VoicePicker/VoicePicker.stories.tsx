import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { fn, userEvent, within } from "storybook/test";
import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { VoicePicker } from "./VoicePicker";

const SAMPLES: VoiceSample[] = [
  {
    label: "Punchy and bold",
    text: "Stop guessing what your buyers want. Ask them, ship it, and measure what moves.",
  },
  {
    label: "Warm and story-led",
    text: "Every campaign starts with a person, not a product.\nSo we begin with the customer's day and work back to the offer.",
  },
];

const meta = {
  title: "Organisms/VoicePicker",
  component: VoicePicker,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-full max-w-xl">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "Asks how an expert should write: two preset voice samples plus a 'paste your own' option, as one radio group. Submit stays disabled until a preset is picked or custom text is entered. Used when hiring an expert and in the raise flow, where the labels and selected card take the expert's colour.",
      },
    },
  },
  args: {
    name: "Maria",
    samples: SAMPLES,
    onPick: fn(),
    onSkip: fn(),
  },
} satisfies Meta<typeof VoicePicker>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithoutName: Story = {
  args: { name: undefined },
};

export const PresetSelected: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(within(canvasElement).getByText("Punchy and bold"));
  },
};

export const CustomSample: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.type(
      within(canvasElement).getByRole("textbox", {
        name: "Custom voice sample",
      }),
      "Keep it breezy, short sentences, no jargon.",
    );
  },
};

export const Submitting: Story = {
  args: { isSubmitting: true },
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByText("Warm and story-led"),
    );
  },
};

export const HiddenHeader: Story = {
  args: { hideHeader: true },
};

export const Compact: Story = {
  args: { compact: true },
};

export const ExpertColour: Story = {
  args: {
    hideHeader: true,
    labelClassName: "text-green-700",
    cardColors: {
      selected: "border-green-400 bg-green-50 ring-2 ring-green-200",
      interactive: "hover:border-green-300 focus-within:ring-green-200",
    },
  },
  play: async ({ canvasElement }) => {
    await userEvent.click(within(canvasElement).getByText("Punchy and bold"));
  },
};

export const SingleSample: Story = {
  args: { samples: SAMPLES.slice(0, 1) },
};
