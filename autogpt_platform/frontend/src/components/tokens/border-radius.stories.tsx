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

// Border radius scale data with rem and px values
// https://tailwindcss.com/docs/border-radius
const borderRadiusScale = [
  { name: "none", value: "0px", rem: "0rem", px: "0px", class: "rounded-none" },
  {
    name: "sm",
    value: "0.125rem",
    rem: "0.125rem",
    px: "2px",
    class: "rounded-sm",
  },
  {
    name: "DEFAULT",
    value: "0.25rem",
    rem: "0.25rem",
    px: "4px",
    class: "rounded",
  },
  {
    name: "md",
    value: "0.375rem",
    rem: "0.375rem",
    px: "6px",
    class: "rounded-md",
  },
  {
    name: "lg",
    value: "0.5rem",
    rem: "0.5rem",
    px: "8px",
    class: "rounded-lg",
  },
  {
    name: "xl",
    value: "0.75rem",
    rem: "0.75rem",
    px: "12px",
    class: "rounded-xl",
  },
  { name: "2xl", value: "1rem", rem: "1rem", px: "16px", class: "rounded-2xl" },
  {
    name: "3xl",
    value: "1.5rem",
    rem: "1.5rem",
    px: "24px",
    class: "rounded-3xl",
  },
  {
    name: "full",
    value: "9999px",
    rem: "9999px",
    px: "9999px",
    class: "rounded-full",
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
            Our border radius system uses a consistent scale to create visual
            hierarchy and maintain design consistency across all components.
            From subtle rounded corners to fully circular elements.
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
                  Apply radius to specific corners or sides
                </Text>
                <div className="space-y-1 font-mono text-sm text-zinc-800">
                  <div>rounded-t-lg → top corners</div>
                  <div>rounded-r-lg → right corners</div>
                  <div>rounded-b-lg → bottom corners</div>
                  <div>rounded-l-lg → left corners</div>
                </div>
              </div>
              <Text variant="body" className="mb-4 text-zinc-600">
                We follow Tailwind CSS border radius system, which provides a
                comprehensive set of radius values for different use cases.
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
            Complete Border Radius Scale
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            All available border radius values in our design system. Each value
            can be applied to all corners or specific corners/sides.
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
          code={`// Border radius examples
<div className="rounded-none">No rounding (0px)</div>
<div className="rounded-sm">Small rounding (2px)</div>
<div className="rounded">Default rounding (4px)</div>
<div className="rounded-md">Medium rounding (6px)</div>
<div className="rounded-lg">Large rounding (8px)</div>
<div className="rounded-xl">Extra large rounding (12px)</div>
<div className="rounded-2xl">2X large rounding (16px)</div>
<div className="rounded-3xl">3X large rounding (24px)</div>
<div className="rounded-full">Fully rounded (circular)</div>

// Directional rounding
<div className="rounded-t-lg">Top corners only</div>
<div className="rounded-r-lg">Right corners only</div>
<div className="rounded-b-lg">Bottom corners only</div>
<div className="rounded-l-lg">Left corners only</div>

// Individual corners
<div className="rounded-tl-lg">Top-left corner</div>
<div className="rounded-tr-lg">Top-right corner</div>
<div className="rounded-bl-lg">Bottom-left corner</div>
<div className="rounded-br-lg">Bottom-right corner</div>`}
        />
      </div>
    </div>
  );
}
