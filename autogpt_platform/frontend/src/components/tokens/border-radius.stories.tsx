import { Text } from "@/components/atoms/Text/Text";
import type { Meta } from "@storybook/nextjs";
import { SquareArrowOutUpRight } from "lucide-react";
import { StoryCode } from "./helpers/StoryCode";

const meta: Meta = {
  title: "Tokens /Border Radius",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

// Border radius scale data based on Figma design tokens
// Custom naming convention: xs, s, m, l, xl, 2xl, full
const borderRadiusScale = [
  {
    name: "xs",
    value: "0.25rem",
    rem: "0.25rem",
    px: "4px",
    class: "rounded-xs",
    description: "Extra small - for subtle rounding",
  },
  {
    name: "s",
    value: "0.5rem",
    rem: "0.5rem",
    px: "8px",
    class: "rounded-s",
    description: "Small - for cards and containers",
  },
  {
    name: "m",
    value: "0.75rem",
    rem: "0.75rem",
    px: "12px",
    class: "rounded-m",
    description: "Medium - for buttons and inputs",
  },
  {
    name: "l",
    value: "1rem",
    rem: "1rem",
    px: "16px",
    class: "rounded-l",
    description: "Large - for panels and modals",
  },
  {
    name: "xl",
    value: "1.25rem",
    rem: "1.25rem",
    px: "20px",
    class: "rounded-xl",
    description: "Extra large - for hero sections",
  },
  {
    name: "2xl",
    value: "1.5rem",
    rem: "1.5rem",
    px: "24px",
    class: "rounded-2xl",
    description: "2X large - for major containers",
  },
  {
    name: "full",
    value: "9999px",
    rem: "9999px",
    px: "9999px",
    class: "rounded-full",
    description: "Full - for pill buttons and circular elements",
  },
];

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
            Our border radius system uses a simplified naming convention (xs, s,
            m, l, xl, 2xl, full) based on our Figma design tokens. This creates
            visual hierarchy and maintains design consistency across all
            components.
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
              <div className="rounded-lg border border-gray-200 p-4">
                <a
                  href="https://tailwindcss.com/docs/border-radius"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mb-2 inline-flex flex-row items-center gap-1 text-base font-semibold text-blue-600 hover:underline"
                >
                  Border Radius Classes{" "}
                  <SquareArrowOutUpRight className="inline-block h-3 w-3" />
                </a>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Used to round the corners of elements
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  rounded-lg → border-radius: 0.5rem (8px)
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-4">
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
                  <div>rounded-t-m → top corners</div>
                  <div>rounded-r-m → right corners</div>
                  <div>rounded-b-m → bottom corners</div>
                  <div>rounded-l-m → left corners</div>
                </div>
              </div>
              <Text variant="body" className="mb-4 text-zinc-600">
                We use a custom border radius system based on our Figma design
                tokens, with simplified naming (xs, s, m, l, xl, 2xl, full) that
                provides consistent radius values optimized for our design
                system.
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
            Design System Border Radius Tokens
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            All border radius values from our Figma design tokens. Each value
            can be applied to all corners or specific corners/sides using our
            simplified naming convention.
          </Text>
        </div>

        <div className="space-y-4">
          {borderRadiusScale.map((radius) => (
            <div
              key={radius.name}
              className="flex items-center rounded-lg border border-gray-200 p-4"
            >
              <div className="flex w-32 flex-col">
                <Text variant="body-medium" className="font-mono text-zinc-800">
                  {radius.name}
                </Text>
                <Text variant="small" className="font-mono text-zinc-500">
                  {radius.class}
                </Text>
              </div>
              <div className="flex w-32 flex-col text-right">
                <p className="font-mono text-xs text-zinc-500">{radius.rem}</p>
                <p className="font-mono text-xs text-zinc-500">{radius.px}</p>
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

        <StoryCode
          code={`// Border radius examples - Design System Tokens
<div className="rounded-xs">Extra small rounding (4px)</div>
<div className="rounded-s">Small rounding (8px)</div>
<div className="rounded-m">Medium rounding (12px)</div>
<div className="rounded-l">Large rounding (16px)</div>
<div className="rounded-xl">Extra large rounding (20px)</div>
<div className="rounded-2xl">2X large rounding (24px)</div>
<div className="rounded-full">Pill buttons (circular)</div>

// Directional rounding (works with all sizes)
<div className="rounded-t-m">Top corners only</div>
<div className="rounded-r-m">Right corners only</div>
<div className="rounded-b-m">Bottom corners only</div>
<div className="rounded-l-m">Left corners only</div>

// Individual corners
<div className="rounded-tl-m">Top-left corner</div>
<div className="rounded-tr-m">Top-right corner</div>
<div className="rounded-bl-m">Bottom-left corner</div>
<div className="rounded-br-m">Bottom-right corner</div>

// Usage recommendations
<button className="rounded-full">Pill Button</button>
<div className="rounded-m">Card Container</div>
<input className="rounded-s">Input Field</input>`}
        />
      </div>
    </div>
  );
}
