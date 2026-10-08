import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Text } from "@/components/atoms/Text/Text";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "./Carousel";

const SLIDES = ["Weekly digest", "Lead enrichment", "Support triage"];

function Slide({ label }: { label: string }) {
  return (
    <div className="flex h-40 items-center justify-center rounded-xl border border-border bg-card">
      <Text variant="body-medium" as="span">
        {label}
      </Text>
    </div>
  );
}

const meta = {
  title: "Molecules/Carousel",
  component: Carousel,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "Embla carousel. `CarouselContent` holds `CarouselItem`s; `CarouselPrevious` and `CarouselNext` are absolutely positioned outside the track by default and can be made `static` to sit in a header. Pass Embla `opts` (`loop`, `align`, `containScroll`) and `setApi` to drive it.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="w-96 px-12">
        <Story />
      </div>
    ),
  ],
  args: {
    children: null,
  },
} satisfies Meta<typeof Carousel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: (args) => (
    <Carousel {...args}>
      <CarouselContent>
        {SLIDES.map((label) => (
          <CarouselItem key={label}>
            <Slide label={label} />
          </CarouselItem>
        ))}
      </CarouselContent>
      <CarouselPrevious />
      <CarouselNext />
    </Carousel>
  ),
};

export const MultipleVisible: Story = {
  render: (args) => (
    <Carousel {...args} opts={{ align: "start", containScroll: "trimSnaps" }}>
      <CarouselContent>
        {[...SLIDES, "Blog drafter", "Inbox triage"].map((label) => (
          <CarouselItem key={label} className="basis-1/2">
            <Slide label={label} />
          </CarouselItem>
        ))}
      </CarouselContent>
      <CarouselPrevious />
      <CarouselNext />
    </Carousel>
  ),
};

export const ControlsInHeader: Story = {
  render: (args) => (
    <Carousel {...args} opts={{ loop: true }}>
      <div className="mb-3 flex items-center justify-between">
        <Text variant="small-medium" as="span" tone="muted">
          Hand-picked
        </Text>
        <div className="flex items-center gap-2">
          <CarouselPrevious className="static" />
          <CarouselNext className="static" />
        </div>
      </div>
      <CarouselContent>
        {SLIDES.map((label) => (
          <CarouselItem key={label}>
            <Slide label={label} />
          </CarouselItem>
        ))}
      </CarouselContent>
    </Carousel>
  ),
};
