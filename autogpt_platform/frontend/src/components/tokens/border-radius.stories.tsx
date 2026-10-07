import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import type { Meta } from "@storybook/nextjs-vite";
import { useEffect, useRef, useState } from "react";
import { StoryCode } from "./helpers/StoryCode";
import { formatPx, getBorderRadiusScale, utilityClass } from "./helpers/theme";

const meta: Meta = {
  title: "Tokens /Border Radius",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
    a11y: { test: "error" },
  },
};

export default meta;

const borderRadiusScale = getBorderRadiusScale();

interface MeasuredRadiusProps {
  value: string;
}

function MeasuredRadius({ value }: MeasuredRadiusProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [px, setPx] = useState<string | null>(null);

  useEffect(() => {
    if (ref.current) setPx(getComputedStyle(ref.current).borderTopLeftRadius);
  }, [value]);

  return (
    <span ref={ref} style={{ borderRadius: value }}>
      {px}
    </span>
  );
}

const radiusCode = [
  "// Every radius token in the resolved theme",
  ...borderRadiusScale.map(
    (radius) =>
      `<div className="${utilityClass("rounded", radius.name)}" /> // ${radius.value}`,
  ),
  "",
  "// Directional rounding works with every token",
  '<div className="rounded-t-xl">Top corners only</div>',
  '<div className="rounded-r-xl">Right corners only</div>',
  '<div className="rounded-tl-xl">Top-left corner only</div>',
].join("\n");

export function AllVariants() {
  return (
    <div className="space-y-12">
      {/* Border Radius System Documentation */}
      <div className="space-y-8">
        <div>
          <Text variant="h1" className="mb-4 text-zinc-800">
            Border Radius
          </Text>
          <Text variant="large" className="text-zinc-600">
            Border radius tokens create visual hierarchy and keep corner
            rounding consistent across components. Every value on this page is
            read from the <code>@theme</code> blocks of{" "}
            <code>src/app/globals.css</code> and Tailwind&apos;s default theme,
            so it always matches the stylesheet.
          </Text>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              Tailwind utilities
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-zinc-200 p-4">
                <a
                  href="https://tailwindcss.com/docs/border-radius"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mb-2 inline-flex flex-row items-center gap-1 text-base font-semibold text-blue-600 hover:underline"
                >
                  Border Radius Classes{" "}
                  <Icon icon={LinkSquare02Icon} size={12} aria-hidden />
                </a>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Used to round the corners of elements
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  rounded-lg → border-radius: 0.5rem (8px)
                </div>
              </div>
              <div className="rounded-lg border border-zinc-200 p-4">
                <Text
                  variant="body-medium"
                  className="mb-2 font-semibold text-zinc-800"
                >
                  Directional Classes
                </Text>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Apply radius to specific corners or sides using our design
                  tokens
                </Text>
                <div className="space-y-1 font-mono text-sm text-zinc-800">
                  <div>rounded-t-xl → top corners</div>
                  <div>rounded-r-xl → right corners</div>
                  <div>rounded-b-xl → bottom corners</div>
                  <div>rounded-l-xl → left corners</div>
                </div>
              </div>
              <Text variant="body" className="mb-4 text-zinc-600">
                Use the tokens listed below rather than arbitrary values such as{" "}
                <code>rounded-[18px]</code>. Tokens marked Custom are added or
                changed by globals.css; the rest are Tailwind defaults.
              </Text>
            </div>
          </div>

          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              FAQ
            </Text>
            <div className="space-y-4">
              <Text
                variant="h3"
                className="mb-2 text-base font-semibold text-zinc-800"
              >
                🤔 Why use border radius tokens?
              </Text>
              <div className="space-y-3 text-zinc-600">
                <Text variant="body">
                  Always use radius classes instead of arbitrary values.
                  Reasons:
                </Text>
                <ul className="ml-4 list-disc space-y-1 text-sm">
                  <li>Ensures consistent corner rounding across components</li>
                  <li>Creates visual hierarchy through systematic scaling</li>
                  <li>Maintains design cohesion and brand consistency</li>
                  <li>Easier to maintain and update globally</li>
                  <li>Prevents inconsistent corner treatments</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Complete Border Radius Scale */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-2 text-xl font-semibold text-zinc-800"
          >
            Border Radius Tokens
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            Every border radius value in the Tailwind theme. Values defined
            through <code>--radius</code> are measured in the browser. Each
            token can be applied to all corners or to specific corners and
            sides.
          </Text>
        </div>

        <div className="space-y-4">
          {borderRadiusScale.map((radius) => (
            <div
              key={radius.name}
              className="flex items-center rounded-lg border border-zinc-200 p-4"
            >
              <div className="flex w-32 flex-col">
                <Text variant="body-medium" className="font-mono text-zinc-800">
                  {radius.name}
                </Text>
                <Text
                  variant="small"
                  className="font-mono text-muted-foreground"
                >
                  {utilityClass("rounded", radius.name)}
                </Text>
              </div>
              <div className="flex w-48 flex-col text-right">
                <Text
                  variant="small"
                  className="font-mono text-muted-foreground"
                >
                  {radius.value}
                </Text>
                <Text
                  variant="small"
                  className="font-mono text-muted-foreground"
                >
                  {formatPx(radius.px) ?? (
                    <MeasuredRadius value={radius.value} />
                  )}
                </Text>
              </div>
              <div className="flex w-24 justify-center">
                {radius.isCustom ? (
                  <Text variant="label" className="text-purple-600">
                    Custom
                  </Text>
                ) : null}
              </div>
              <div className="ml-8 flex-1">
                <div className="flex items-center gap-4">
                  <div
                    className="h-16 w-16 bg-blue-500"
                    style={{ borderRadius: radius.value }}
                  ></div>
                  <div
                    className="h-12 w-24 bg-green-500"
                    style={{ borderRadius: radius.value }}
                  ></div>
                  <div
                    className="h-8 w-32 bg-purple-500"
                    style={{ borderRadius: radius.value }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <StoryCode code={radiusCode} />
      </div>
    </div>
  );
}
