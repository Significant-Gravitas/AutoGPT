import React from "react";
import { Button } from "./Button";
import type { ButtonProps } from "./helpers";
import { useTrackEvent } from "@/services/feature-flags/use-track-event";

export type TrackedButtonProps = ButtonProps & {
  /**
   * Event key to track when button is clicked
   * If not provided, no tracking will occur
   */
  trackEventKey?: string;
  /**
   * Additional data to send with the tracking event
   */
  trackEventData?: Record<string, any>;
  /**
   * Metric value for numeric metrics (e.g., duration, count, amount)
   */
  trackMetricValue?: number;
};

/**
 * Button component with built-in LaunchDarkly event tracking
 * Extends the base Button atom with tracking capabilities
 */
export function TrackedButton({
  trackEventKey,
  trackEventData,
  trackMetricValue,
  onClick,
  children,
  ...props
}: TrackedButtonProps) {
  const { track } = useTrackEvent();

  const handleClick = (
    e: React.MouseEvent<HTMLButtonElement | HTMLAnchorElement>,
  ) => {
    // Track the event if configured
    if (trackEventKey) {
      track(
        trackEventKey,
        {
          buttonLabel:
            typeof children === "string"
              ? children.substring(0, 100)
              : undefined,
          timestamp: new Date().toISOString(),
          ...trackEventData,
        },
        trackMetricValue,
      );
    }

    // Call the original onClick handler
    if (onClick) {
      if (props.as === "NextLink") {
        (onClick as React.MouseEventHandler<HTMLAnchorElement>)(
          e as React.MouseEvent<HTMLAnchorElement>,
        );
      } else {
        (onClick as React.MouseEventHandler<HTMLButtonElement>)(
          e as React.MouseEvent<HTMLButtonElement>,
        );
      }
    }
  };

  return <Button {...props} onClick={handleClick}>{children}</Button>;
}

// Export common tracked button presets for consistency
export const TrackedPrimaryButton: React.FC<TrackedButtonProps> = (props) => (
  <TrackedButton variant="primary" {...props} />
);

export const TrackedSecondaryButton: React.FC<TrackedButtonProps> = (
  props,
) => <TrackedButton variant="secondary" {...props} />;

export const TrackedDestructiveButton: React.FC<TrackedButtonProps> = (
  props,
) => <TrackedButton variant="destructive" {...props} />;

export const TrackedGhostButton: React.FC<TrackedButtonProps> = (props) => (
  <TrackedButton variant="ghost" {...props} />
);