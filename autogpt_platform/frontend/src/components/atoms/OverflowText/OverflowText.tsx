import { Text, type TextProps } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

interface Props extends Omit<TextProps, "children"> {
  value: string | ReactNode;
  /** How text that does not fit ends: cut with "…", or faded out at the
   *  right edge with no ellipsis. Either way the full text is in a tooltip. */
  overflow?: "ellipsis" | "fade";
}

export function OverflowText({
  value,
  className,
  variant = "body",
  overflow = "ellipsis",
  ...restProps
}: Props) {
  const elementRef = useRef<HTMLSpanElement | null>(null);
  const [isTruncated, setIsTruncated] = useState(false);

  function updateTruncation() {
    const element = elementRef.current;

    if (!element) {
      return;
    }

    const hasOverflow = element.scrollWidth > element.clientWidth;

    setIsTruncated(hasOverflow);
  }

  function setupResizeListener() {
    function handleResize() {
      updateTruncation();
    }

    window.addEventListener("resize", handleResize);

    return function cleanupResizeListener() {
      window.removeEventListener("resize", handleResize);
    };
  }

  function setupObserver() {
    const element = elementRef.current;

    if (!element || typeof ResizeObserver === "undefined") {
      return undefined;
    }

    function handleResizeObserver() {
      updateTruncation();
    }

    const observer = new ResizeObserver(handleResizeObserver);

    observer.observe(element);

    return function disconnectObserver() {
      observer.disconnect();
    };
  }

  useEffect(() => {
    if (typeof value === "string") updateTruncation();
  }, [value]);

  useEffect(setupResizeListener, []);
  useEffect(setupObserver, []);

  const content = (
    <span
      ref={elementRef}
      data-slot="overflow-text"
      className={cn(
        "block min-w-0 overflow-hidden whitespace-nowrap",
        overflow === "ellipsis" && "text-ellipsis",
        // Only while it is cut: text that fits keeps its last letters whole.
        overflow === "fade" && isTruncated && "mask-fade-right",
      )}
    >
      <Text
        data-slot="overflow-text-label"
        variant={variant}
        as="span"
        className={className}
        {...restProps}
      >
        {value}
      </Text>
    </span>
  );

  if (isTruncated) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{content}</TooltipTrigger>
          <TooltipContent data-slot="overflow-text-tooltip">
            {typeof value === "string" ? (
              <p data-slot="overflow-text-tooltip-text">{value}</p>
            ) : (
              value
            )}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return content;
}
