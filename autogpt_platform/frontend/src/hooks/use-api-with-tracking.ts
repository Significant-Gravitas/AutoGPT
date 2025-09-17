"use client";

import { useCallback } from "react";
import {
  useTrackEvent,
  EventKeys,
} from "@/services/feature-flags/use-track-event";

/**
 * Hook that wraps API calls with automatic error tracking
 * Tracks API errors, response times, and success rates
 */
export function useAPIWithTracking() {
  const { track } = useTrackEvent();

  /**
   * Wraps an API call with tracking
   * @param apiCall The async API function to execute
   * @param eventName Optional custom event name for this API call
   * @param metadata Additional metadata to track
   */
  const trackAPICall = useCallback(
    async <T>(
      apiCall: () => Promise<T>,
      eventName?: string,
      metadata?: Record<string, any>,
    ): Promise<T> => {
      const startTime = performance.now();

      try {
        const result = await apiCall();
        const duration = performance.now() - startTime;

        // Track successful API response time
        if (duration > 1000) {
          // Only track if response took more than 1 second
          track(
            EventKeys.API_RESPONSE_TIME,
            {
              eventName,
              success: true,
              ...metadata,
            },
            Math.round(duration),
          );
        }

        return result;
      } catch (error) {
        const duration = performance.now() - startTime;

        // Track API error
        track(EventKeys.API_ERROR, {
          eventName,
          errorMessage: error instanceof Error ? error.message : String(error),
          errorCode: (error as any)?.code || (error as any)?.status,
          duration: Math.round(duration),
          timestamp: new Date().toISOString(),
          ...metadata,
        });

        // Re-throw the error so the calling code can handle it
        throw error;
      }
    },
    [track],
  );

  /**
   * Tracks validation errors separately from API errors
   */
  const trackValidationError = useCallback(
    (fieldName: string, errorMessage: string, formName?: string) => {
      track(EventKeys.VALIDATION_ERROR, {
        fieldName,
        errorMessage,
        formName,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  /**
   * Tracks generic errors that aren't API or validation related
   */
  const trackError = useCallback(
    (error: Error | unknown, context?: string) => {
      track(EventKeys.ERROR_OCCURRED, {
        errorMessage: error instanceof Error ? error.message : String(error),
        errorStack: error instanceof Error ? error.stack : undefined,
        context,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  return {
    trackAPICall,
    trackValidationError,
    trackError,
  };
}

/**
 * Higher-order function to wrap async functions with error tracking
 * Useful for wrapping server actions or API client methods
 */
export function withAPITracking<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  eventName: string,
  getMetadata?: (...args: Parameters<T>) => Record<string, any>,
): T {
  return (async (...args: Parameters<T>) => {
    const startTime = performance.now();

    try {
      const result = await fn(...args);
      return result;
    } catch (error) {
      // Since this is a wrapper, we can't use hooks directly
      // Log to console for now - in production, this could send to a logging service
      console.error(`API Error in ${eventName}:`, {
        error: error instanceof Error ? error.message : error,
        metadata: getMetadata ? getMetadata(...args) : undefined,
        duration: performance.now() - startTime,
      });

      throw error;
    }
  }) as T;
}
