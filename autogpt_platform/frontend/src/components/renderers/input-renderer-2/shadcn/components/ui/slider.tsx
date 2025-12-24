"use client";

import { Range, Root, Thumb, Track } from "@radix-ui/react-slider";
import { ComponentProps, useMemo } from "react";

import { cn } from "../../lib/utils";

/**
 * A slider component for selecting a numeric value within a range
 *
 * @param props - The props for the Slider component
 * @param props.className - Additional CSS classes to apply to the slider
 * @param ref - The forwarded ref for the slider element
 * @returns A slider input element with track and thumb
 */
function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  ...props
}: ComponentProps<typeof Root>) {
  const _values = useMemo(
    () =>
      Array.isArray(value)
        ? value
        : Array.isArray(defaultValue)
          ? defaultValue
          : [min, max],
    [value, defaultValue, min, max],
  );
  return (
    <Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      className={cn(
        "relative flex w-full touch-none select-none items-center data-[orientation=vertical]:h-full data-[orientation=vertical]:min-h-44 data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col data-[disabled]:opacity-50",
        className,
      )}
      {...props}
    >
      <Track
        data-slot="slider-track"
        className={cn(
          "relative grow overflow-hidden rounded-full bg-muted data-[orientation=horizontal]:h-1.5 data-[orientation=vertical]:h-full data-[orientation=horizontal]:w-full data-[orientation=vertical]:w-1.5",
        )}
      >
        <Range
          data-slot="slider-range"
          className={cn(
            "absolute bg-primary data-[orientation=horizontal]:h-full data-[orientation=vertical]:w-full",
          )}
        />
      </Track>
      {Array.from({ length: _values.length }, (_, index) => (
        <Thumb
          data-slot="slider-thumb"
          key={index}
          className="focus-visible:outline-hidden block size-4 shrink-0 rounded-full border border-primary bg-background shadow-sm ring-ring/50 transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 disabled:pointer-events-none disabled:opacity-50"
        />
      ))}
    </Root>
  );
}

export { Slider };
