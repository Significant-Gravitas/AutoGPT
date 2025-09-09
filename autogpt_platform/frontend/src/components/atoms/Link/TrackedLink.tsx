import { forwardRef } from "react";
import { Link } from "./Link";
import { useTrackEvent } from "@/services/feature-flags/use-track-event";

interface TrackedLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  isExternal?: boolean;
  variant?: "primary" | "secondary";
  /**
   * Event key to track when link is clicked
   */
  trackEventKey?: string;
  /**
   * Additional data to send with the tracking event
   */
  trackEventData?: Record<string, any>;
  /**
   * Metric value for numeric metrics
   */
  trackMetricValue?: number;
}

/**
 * Link component with built-in LaunchDarkly event tracking
 * Extends the base Link atom with tracking capabilities
 */
const TrackedLink = forwardRef<HTMLAnchorElement, TrackedLinkProps>(
  function TrackedLink(
    {
      trackEventKey,
      trackEventData,
      trackMetricValue,
      children,
      href,
      ...props
    },
    ref,
  ) {
    const { track } = useTrackEvent();

    const handleClick = () => {
      if (trackEventKey) {
        track(
          trackEventKey,
          {
            href,
            linkText:
              typeof children === "string"
                ? children.substring(0, 100)
                : undefined,
            isExternal: props.isExternal || false,
            timestamp: new Date().toISOString(),
            ...trackEventData,
          },
          trackMetricValue,
        );
      }
    };

    // Wrap the Link in a span to capture clicks
    return (
      <span onClick={handleClick} style={{ display: "inline" }}>
        <Link ref={ref} href={href} {...props}>
          {children}
        </Link>
      </span>
    );
  },
);

export { TrackedLink };
export type { TrackedLinkProps };
