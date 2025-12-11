import React, { useEffect, useRef, useState } from "react";
import { ArrowLeftIcon, ArrowRightIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

interface HorizontalScrollAreaProps {
  children: React.ReactNode;
  wrapperClassName?: string;
  scrollContainerClassName?: string;
  scrollAmount?: number;
  dependencyList?: React.DependencyList;
}

const defaultDependencies: React.DependencyList = [];
const baseScrollClasses =
  "flex gap-2 overflow-x-auto px-8 [scrollbar-width:none] [-ms-overflow-style:'none'] [&::-webkit-scrollbar]:hidden";

export const HorizontalScroll: React.FC<HorizontalScrollAreaProps> = ({
  children,
  wrapperClassName,
  scrollContainerClassName,
  scrollAmount = 300,
  dependencyList = defaultDependencies,
}) => {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const scrollByDelta = (delta: number) => {
    if (!scrollRef.current) {
      return;
    }
    scrollRef.current.scrollBy({ left: delta, behavior: "smooth" });
  };

  const updateScrollState = () => {
    const element = scrollRef.current;
    if (!element) {
      setCanScrollLeft(false);
      setCanScrollRight(false);
      return;
    }
    setCanScrollLeft(element.scrollLeft > 0);
    setCanScrollRight(
      Math.ceil(element.scrollLeft + element.clientWidth) < element.scrollWidth,
    );
  };

  useEffect(() => {
    updateScrollState();
    const element = scrollRef.current;
    if (!element) {
      return;
    }
    const handleScroll = () => updateScrollState();
    element.addEventListener("scroll", handleScroll);
    window.addEventListener("resize", handleScroll);
    return () => {
      element.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    };
  }, dependencyList);

  return (
    <div className={wrapperClassName}>
      <div className="group relative">
        <div
          ref={scrollRef}
          className={cn(baseScrollClasses, scrollContainerClassName)}
        >
          {children}
        </div>
        {canScrollLeft && (
          <div className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-white via-white/80 to-white/0" />
        )}
        {canScrollRight && (
          <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-white via-white/80 to-white/0" />
        )}
        {canScrollLeft && (
          <button
            type="button"
            aria-label="Scroll left"
            className="pointer-events-none absolute left-2 top-5 -translate-y-1/2 opacity-0 transition-opacity duration-200 group-hover:pointer-events-auto group-hover:opacity-100"
            onClick={() => scrollByDelta(-scrollAmount)}
          >
            <ArrowLeftIcon
              size={28}
              className="rounded-full bg-zinc-700 p-1 text-white drop-shadow"
              weight="light"
            />
          </button>
        )}
        {canScrollRight && (
          <button
            type="button"
            aria-label="Scroll right"
            className="pointer-events-none absolute right-2 top-5 -translate-y-1/2 opacity-0 transition-opacity duration-200 group-hover:pointer-events-auto group-hover:opacity-100"
            onClick={() => scrollByDelta(scrollAmount)}
          >
            <ArrowRightIcon
              size={28}
              className="rounded-full bg-zinc-700 p-1 text-white drop-shadow"
              weight="light"
            />
          </button>
        )}
      </div>
    </div>
  );
};
