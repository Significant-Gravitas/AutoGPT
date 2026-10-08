"use client";

// Kobra's Embla carousel, exposed for app code (which may not import
// `@/components/ui/*`). Arrow keys scroll, RTL is handled, and
// Previous/Next disable at the ends unless `opts.loop` is set.
export {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
