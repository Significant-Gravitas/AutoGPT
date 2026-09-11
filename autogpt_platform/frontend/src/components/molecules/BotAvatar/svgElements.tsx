import type { SVGMotionProps } from "framer-motion";
import { createElement, forwardRef, type ElementType } from "react";

// The avatar parts draw with motion elements when they're alive in the
// browser and with plain SVG elements when rendered to a static string
// (the /avatars/[file] route runs in the server layer, where framer-motion
// and hooks are off limits). Parts take the element set as a prop so they
// never import framer-motion themselves.
export interface SvgEls {
  path: ElementType<SVGMotionProps<SVGPathElement>>;
  g: ElementType<SVGMotionProps<SVGGElement>>;
  ellipse: ElementType<SVGMotionProps<SVGEllipseElement>>;
  circle: ElementType<SVGMotionProps<SVGCircleElement>>;
}

const MOTION_ONLY_PROPS = new Set([
  "initial",
  "animate",
  "exit",
  "transition",
  "variants",
  "whileHover",
  "whileTap",
  "whileFocus",
  "whileDrag",
  "whileInView",
  "layout",
  "layoutId",
  "onAnimationStart",
  "onAnimationComplete",
  "onUpdate",
]);
const ATTRIBUTE_TARGETS = new Set([
  "d",
  "rx",
  "ry",
  "r",
  "cx",
  "cy",
  "opacity",
]);

function isPlainTarget(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function transformFrom(target: Record<string, unknown>): string | undefined {
  const num = (key: string, fallback: number) =>
    typeof target[key] === "number" ? (target[key] as number) : fallback;
  const parts: string[] = [];
  const x = num("x", 0);
  const y = num("y", 0);
  if (x || y) parts.push(`translate(${x} ${y})`);
  const rotate = num("rotate", 0);
  if (rotate) parts.push(`rotate(${rotate})`);
  const scale = num("scale", 1);
  const scaleX = num("scaleX", scale);
  const scaleY = num("scaleY", scale);
  if (scaleX !== 1 || scaleY !== 1) parts.push(`scale(${scaleX} ${scaleY})`);
  return parts.length ? parts.join(" ") : undefined;
}

// A motion element's resting frame: the `animate` target becomes plain
// attributes (or a transform), and every motion-only prop is dropped.
function staticEl<T extends SVGElement>(tag: string) {
  return forwardRef<T, SVGMotionProps<T>>(function StaticSvgEl(props, ref) {
    const attrs: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(props)) {
      if (!MOTION_ONLY_PROPS.has(key)) attrs[key] = value;
    }
    if (isPlainTarget(props.animate)) {
      for (const [key, value] of Object.entries(props.animate)) {
        if (ATTRIBUTE_TARGETS.has(key) && !Array.isArray(value)) {
          attrs[key] = value;
        }
      }
      const transform = transformFrom(props.animate);
      if (transform) {
        attrs.transform = [attrs.transform, transform]
          .filter(Boolean)
          .join(" ");
      }
    }
    return createElement(tag, { ...attrs, ref });
  });
}

export const STATIC_ELS: SvgEls = {
  path: staticEl<SVGPathElement>("path"),
  g: staticEl<SVGGElement>("g"),
  ellipse: staticEl<SVGEllipseElement>("ellipse"),
  circle: staticEl<SVGCircleElement>("circle"),
};
