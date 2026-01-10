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
}

export function OverflowText(props: Props) {
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
    if (typeof props.value === "string") updateTruncation();
  }, [props.value]);

  useEffect(setupResizeListener, []);
  useEffect(setupObserver, []);

  const { value, className, variant = "body", ...restProps } = props;

  const content = (
    <span
      ref={elementRef}
      className={cn(
        "block min-w-0 overflow-hidden text-ellipsis whitespace-nowrap",
      )}
    >
      <Text variant={variant} className={className} {...restProps}>
        {value}
      </Text>
    </span>
  );

  if (isTruncated) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{content}</TooltipTrigger>
          <TooltipContent>
            {typeof value === "string" ? <p>{value}</p> : value}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return content;
}
