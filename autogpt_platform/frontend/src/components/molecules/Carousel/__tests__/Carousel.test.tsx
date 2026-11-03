import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { createRef } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Carousel, CarouselContent, CarouselItem } from "../Carousel";

const { api, useEmblaCarousel } = vi.hoisted(() => {
  const api = { scrollPrev: vi.fn(), scrollNext: vi.fn() };
  const emblaRef = vi.fn();
  return { api, useEmblaCarousel: vi.fn(() => [emblaRef, api]) };
});

vi.mock("embla-carousel-react", () => ({ default: useEmblaCarousel }));

function Slides(props: React.ComponentProps<typeof Carousel>) {
  return (
    <Carousel {...props}>
      <CarouselContent>
        <CarouselItem>One</CarouselItem>
        <CarouselItem>Two</CarouselItem>
      </CarouselContent>
    </Carousel>
  );
}

describe("Carousel", () => {
  beforeEach(() => {
    api.scrollPrev.mockClear();
    api.scrollNext.mockClear();
    useEmblaCarousel.mockClear();
  });

  it("renders a carousel region with one slide group per item", () => {
    render(<Slides />);
    expect(
      screen.getByRole("region").getAttribute("aria-roledescription"),
    ).toBe("carousel");
    const slides = screen.getAllByRole("group");
    expect(slides).toHaveLength(2);
    expect(slides[0].getAttribute("aria-roledescription")).toBe("slide");
    expect(slides[0].classList.contains("pl-4")).toBe(true);
  });

  it("moves with the arrow keys", () => {
    render(<Slides />);
    const region = screen.getByRole("region");

    fireEvent.keyDown(region, { key: "ArrowRight" });
    expect(api.scrollNext).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(region, { key: "ArrowLeft" });
    expect(api.scrollPrev).toHaveBeenCalledTimes(1);
  });

  it("ignores arrow keys while an IME is composing", () => {
    render(<Slides />);
    fireEvent.keyDown(screen.getByRole("region"), {
      key: "ArrowRight",
      isComposing: true,
    });
    expect(api.scrollNext).not.toHaveBeenCalled();
  });

  it("passes the axis to embla and hands the api to setApi", () => {
    const setApi = vi.fn();
    render(
      <Slides orientation="vertical" setApi={setApi} opts={{ loop: true }} />,
    );
    expect(useEmblaCarousel).toHaveBeenCalledWith(
      { loop: true, axis: "y" },
      undefined,
    );
    expect(setApi).toHaveBeenCalledWith(api);
    expect(screen.getAllByRole("group")[0].classList.contains("pt-4")).toBe(
      true,
    );
  });

  it("forwards refs to the region, the slide track and each slide", () => {
    const root = createRef<HTMLDivElement>();
    const track = createRef<HTMLDivElement>();
    const slide = createRef<HTMLDivElement>();
    render(
      <Carousel ref={root}>
        <CarouselContent ref={track}>
          <CarouselItem ref={slide}>One</CarouselItem>
        </CarouselContent>
      </Carousel>,
    );
    expect(root.current).toBe(screen.getByRole("region"));
    expect(track.current?.getAttribute("data-slot")).toBe("carousel-content");
    expect(slide.current).toBe(screen.getByRole("group"));
  });

  it("refuses parts used outside a Carousel", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<CarouselItem>Lost</CarouselItem>)).toThrow(
      "useCarousel must be used within a <Carousel />",
    );
  });
});
