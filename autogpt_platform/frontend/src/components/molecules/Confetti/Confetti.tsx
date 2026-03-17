"use client";

import type { ReactNode } from "react";
import React, {
  createContext,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from "react";
import type {
  GlobalOptions as ConfettiGlobalOptions,
  CreateTypes as ConfettiInstance,
  Options as ConfettiOptions,
} from "canvas-confetti";
import confetti from "canvas-confetti";

import * as Sentry from "@sentry/nextjs";
import { cn } from "@/lib/utils";

/** AutoGPT design system purple palette for confetti */
const AGPT_CONFETTI_COLORS = [
  "#7733f5", // purple-500
  "#925cf7", // purple-400
  "#a476f8", // purple-300
  "#c0a1fa", // purple-200
  "#6c2edf", // purple-600
  "#5424ae", // purple-700
  "#efe8fe", // purple-100
];

type Api = {
  fire: (options?: ConfettiOptions) => void;
};

interface Props extends React.ComponentPropsWithRef<"canvas"> {
  options?: ConfettiOptions;
  globalOptions?: ConfettiGlobalOptions;
  manualstart?: boolean;
  children?: ReactNode;
}

export type ConfettiRef = Api | null;

export const ConfettiContext = createContext<Api>({} as Api);

const ConfettiComponent = forwardRef<ConfettiRef, Props>(
  function Confetti(props, ref) {
    const {
      options,
      globalOptions,
      manualstart = false,
      children,
      className,
      ...rest
    } = props;

    const canvasElRef = useRef<HTMLCanvasElement | null>(null);
    const instanceRef = useRef<ConfettiInstance | null>(null);
    const hasAutoStartedRef = useRef(false);

    // Create the confetti instance once after mount via useEffect,
    // so React never re-fires a callback ref on re-renders.
    useEffect(() => {
      const node = canvasElRef.current;
      if (!node || instanceRef.current) return;

      const dpr = window.devicePixelRatio || 1;
      node.width = node.offsetWidth * dpr;
      node.height = node.offsetHeight * dpr;

      instanceRef.current = confetti.create(node, {
        resize: true,
        useWorker: false,
        ...globalOptions,
      });

      return () => {
        instanceRef.current?.reset();
        instanceRef.current = null;
      };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const fire = useMemo(
      () =>
        async (opts: ConfettiOptions = {}) => {
          try {
            await instanceRef.current?.({
              colors: AGPT_CONFETTI_COLORS,
              ...options,
              ...opts,
            });
          } catch (error) {
            Sentry.captureException(error);
          }
        },
      [options],
    );

    const api = useMemo(() => ({ fire }), [fire]);

    useImperativeHandle(ref, () => api, [api]);

    useEffect(() => {
      if (manualstart || hasAutoStartedRef.current) return;
      hasAutoStartedRef.current = true;
      void fire();
    }, [manualstart, fire]);

    return (
      <ConfettiContext.Provider value={api}>
        <canvas
          ref={canvasElRef}
          style={{ width: "100%", height: "100%" }}
          className={cn(
            "pointer-events-none fixed inset-0 z-50 h-full w-full",
            className,
          )}
          {...rest}
        />
        {children}
      </ConfettiContext.Provider>
    );
  },
);

ConfettiComponent.displayName = "Confetti";

export { ConfettiComponent as Confetti, AGPT_CONFETTI_COLORS };
export type { Props as ConfettiProps, ConfettiOptions };
