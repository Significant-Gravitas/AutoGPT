import type { Meta, StoryObj } from "@storybook/nextjs";
import { Carousel, CarouselContent, CarouselItem } from "./Carousel";

const meta: Meta<typeof Carousel> = {
  title: "Molecules/Carousel",
  component: Carousel,
  parameters: { layout: "centered" },
};

export default meta;
type Story = StoryObj<typeof Carousel>;

interface Props {
  count: number;
}

function Slides({ count }: Props) {
  return (
    <CarouselContent>
      {Array.from({ length: count }, (_, i) => (
        <CarouselItem key={i} className="basis-4/5">
          <div className="flex h-40 items-center justify-center rounded-large border border-border-base bg-surface">
            Slide {i + 1}
          </div>
        </CarouselItem>
      ))}
    </CarouselContent>
  );
}

export const Default: Story = {
  render: () => (
    <Carousel className="w-[340px]">
      <Slides count={5} />
    </Carousel>
  ),
};

export const Looping: Story = {
  render: () => (
    <Carousel className="w-[340px]" opts={{ loop: true }}>
      <Slides count={5} />
    </Carousel>
  ),
};

export const Vertical: Story = {
  render: () => (
    <Carousel
      className="w-[340px]"
      orientation="vertical"
      opts={{ align: "start" }}
    >
      <CarouselContent className="h-[200px]">
        {Array.from({ length: 4 }, (_, i) => (
          <CarouselItem key={i} className="basis-1/2">
            <div className="flex h-20 items-center justify-center rounded-large border border-border-base bg-surface">
              Slide {i + 1}
            </div>
          </CarouselItem>
        ))}
      </CarouselContent>
    </Carousel>
  ),
};
