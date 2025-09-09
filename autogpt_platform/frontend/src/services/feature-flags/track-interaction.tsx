"use client";

import React, { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { useTrackEvent } from "./use-track-event";

interface TrackInteractionProps {
  children: React.ReactNode;
  eventKey?: string;
  eventData?: Record<string, any>;
  trackPageView?: boolean;
  trackClicks?: boolean;
  className?: string;
}

/**
 * Component that automatically tracks user interactions
 * Can track page views, clicks, and custom events
 */
export function TrackInteraction({
  children,
  eventKey,
  eventData,
  trackPageView = false,
  trackClicks = false,
  className,
}: TrackInteractionProps) {
  const { track } = useTrackEvent();
  const pathname = usePathname();
  const hasTrackedPageView = useRef(false);

  // Track page view on mount
  useEffect(() => {
    if (trackPageView && !hasTrackedPageView.current) {
      track(`page-viewed-${pathname.replace(/\//g, "-")}`, {
        path: pathname,
        timestamp: new Date().toISOString(),
        ...eventData,
      });
      hasTrackedPageView.current = true;
    }
  }, [trackPageView, pathname, track, eventData]);

  // Handle click tracking
  const handleClick = (e: React.MouseEvent) => {
    if (trackClicks && eventKey) {
      const target = e.target as HTMLElement;
      track(eventKey, {
        elementType: target.tagName.toLowerCase(),
        elementText: target.textContent?.substring(0, 100),
        path: pathname,
        timestamp: new Date().toISOString(),
        ...eventData,
      });
    }
  };

  if (trackClicks) {
    return (
      <div onClick={handleClick} className={className}>
        {children}
      </div>
    );
  }

  return <>{children}</>;
}

/**
 * HOC to add tracking to any component
 */
export function withTracking<P extends object>(
  Component: React.ComponentType<P>,
  defaultEventKey?: string,
  defaultEventData?: Record<string, any>,
) {
  const WrappedComponent = React.forwardRef<any, P>((props, ref) => {
    return (
      <TrackInteraction eventKey={defaultEventKey} eventData={defaultEventData}>
        <Component {...(props as P)} ref={ref} />
      </TrackInteraction>
    );
  });
  WrappedComponent.displayName = `withTracking(${Component.displayName || Component.name || "Component"})`;
  return WrappedComponent;
}

/**
 * Track button component that automatically sends events on click
 */
interface TrackButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  eventKey: string;
  eventData?: Record<string, any>;
  children: React.ReactNode;
}

export function TrackButton({
  eventKey,
  eventData,
  onClick,
  children,
  ...props
}: TrackButtonProps) {
  const { track } = useTrackEvent();

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    track(eventKey, eventData);
    onClick?.(e);
  };

  return (
    <button onClick={handleClick} {...props}>
      {children}
    </button>
  );
}

/**
 * Track link component that automatically sends events on click
 */
interface TrackLinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  eventKey: string;
  eventData?: Record<string, any>;
  children: React.ReactNode;
}

export function TrackLink({
  eventKey,
  eventData,
  onClick,
  children,
  ...props
}: TrackLinkProps) {
  const { track } = useTrackEvent();

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    track(eventKey, {
      href: props.href,
      ...eventData,
    });
    onClick?.(e);
  };

  return (
    <a onClick={handleClick} {...props}>
      {children}
    </a>
  );
}
