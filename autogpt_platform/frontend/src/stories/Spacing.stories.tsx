import type { Meta, StoryObj } from "@storybook/react";
import { Text } from "@/components/_new/Text/Text";
import { StoryCode } from "@/stories/helpers/StoryCode";

const meta: Meta = {
  title: "Design System/ Tokens /Spacing",
  parameters: {
    layout: "fullscreen",
  },
};

export default meta;

// Spacing scale data with rem and px values
const spacingScale = [
  { name: "0", value: "0", rem: "0rem", px: "0px", class: "m-0" },
  { name: "px", value: "1px", rem: "0.0625rem", px: "1px", class: "m-px" },
  {
    name: "0.5",
    value: "0.125rem",
    rem: "0.125rem",
    px: "2px",
    class: "m-0.5",
  },
  { name: "1", value: "0.25rem", rem: "0.25rem", px: "4px", class: "m-1" },
  {
    name: "1.5",
    value: "0.375rem",
    rem: "0.375rem",
    px: "6px",
    class: "m-1.5",
  },
  { name: "2", value: "0.5rem", rem: "0.5rem", px: "8px", class: "m-2" },
  {
    name: "2.5",
    value: "0.625rem",
    rem: "0.625rem",
    px: "10px",
    class: "m-2.5",
  },
  { name: "3", value: "0.75rem", rem: "0.75rem", px: "12px", class: "m-3" },
  {
    name: "3.5",
    value: "0.875rem",
    rem: "0.875rem",
    px: "14px",
    class: "m-3.5",
  },
  { name: "4", value: "1rem", rem: "1rem", px: "16px", class: "m-4" },
  { name: "5", value: "1.25rem", rem: "1.25rem", px: "20px", class: "m-5" },
  { name: "6", value: "1.5rem", rem: "1.5rem", px: "24px", class: "m-6" },
  { name: "7", value: "1.75rem", rem: "1.75rem", px: "28px", class: "m-7" },
  { name: "8", value: "2rem", rem: "2rem", px: "32px", class: "m-8" },
  { name: "9", value: "2.25rem", rem: "2.25rem", px: "36px", class: "m-9" },
  { name: "10", value: "2.5rem", rem: "2.5rem", px: "40px", class: "m-10" },
  { name: "11", value: "2.75rem", rem: "2.75rem", px: "44px", class: "m-11" },
  { name: "12", value: "3rem", rem: "3rem", px: "48px", class: "m-12" },
  { name: "14", value: "3.5rem", rem: "3.5rem", px: "56px", class: "m-14" },
  { name: "16", value: "4rem", rem: "4rem", px: "64px", class: "m-16" },
  { name: "20", value: "5rem", rem: "5rem", px: "80px", class: "m-20" },
  { name: "24", value: "6rem", rem: "6rem", px: "96px", class: "m-24" },
  { name: "28", value: "7rem", rem: "7rem", px: "112px", class: "m-28" },
  { name: "32", value: "8rem", rem: "8rem", px: "128px", class: "m-32" },
  { name: "36", value: "9rem", rem: "9rem", px: "144px", class: "m-36" },
  { name: "40", value: "10rem", rem: "10rem", px: "160px", class: "m-40" },
  { name: "44", value: "11rem", rem: "11rem", px: "176px", class: "m-44" },
  { name: "48", value: "12rem", rem: "12rem", px: "192px", class: "m-48" },
  { name: "52", value: "13rem", rem: "13rem", px: "208px", class: "m-52" },
  { name: "56", value: "14rem", rem: "14rem", px: "224px", class: "m-56" },
  { name: "60", value: "15rem", rem: "15rem", px: "240px", class: "m-60" },
  { name: "64", value: "16rem", rem: "16rem", px: "256px", class: "m-64" },
  { name: "72", value: "18rem", rem: "18rem", px: "288px", class: "m-72" },
  { name: "80", value: "20rem", rem: "20rem", px: "320px", class: "m-80" },
  { name: "96", value: "24rem", rem: "24rem", px: "384px", class: "m-96" },
];

// Common spacing values for quick reference
const commonSpacing = [
  {
    name: "1",
    rem: "0.25rem",
    px: "4px",
    class: "m-1",
    usage: "Tight spacing, small gaps",
  },
  {
    name: "2",
    rem: "0.5rem",
    px: "8px",
    class: "m-2",
    usage: "Small spacing, form elements",
  },
  {
    name: "3",
    rem: "0.75rem",
    px: "12px",
    class: "m-3",
    usage: "Default spacing, buttons",
  },
  {
    name: "4",
    rem: "1rem",
    px: "16px",
    class: "m-4",
    usage: "Medium spacing, cards",
  },
  {
    name: "6",
    rem: "1.5rem",
    px: "24px",
    class: "m-6",
    usage: "Large spacing, sections",
  },
  {
    name: "8",
    rem: "2rem",
    px: "32px",
    class: "m-8",
    usage: "Extra large spacing",
  },
  {
    name: "12",
    rem: "3rem",
    px: "48px",
    class: "m-12",
    usage: "Section dividers",
  },
  {
    name: "16",
    rem: "4rem",
    px: "64px",
    class: "m-16",
    usage: "Page sections",
  },
  {
    name: "24",
    rem: "6rem",
    px: "96px",
    class: "m-24",
    usage: "Major page divisions",
  },
];

export function AllVariants() {
  return (
    <div className="space-y-12">
      {/* Spacing System Documentation */}
      <div className="space-y-8">
        <div>
          <Text variant="h1" className="mb-4 text-zinc-800">
            Spacing System
          </Text>
          <Text variant="large" className="text-zinc-600">
            Our spacing system uses a consistent scale based on rem units to
            ensure proper spacing relationships across all components and
            layouts. The spacing tokens are identical for both margin and
            padding utilities.
          </Text>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <Text variant="h2" className="mb-4 text-zinc-800">
              Spacing Scale
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <Text variant="h3" className="mb-2 text-zinc-800">
                  <a
                    href="https://tailwindcss.com/docs/margin"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    Margin Classes
                  </a>
                </Text>
                <Text variant="small" className="mb-2 text-zinc-600">
                  Used for external spacing between elements
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  m-4 → margin: 1rem (16px)
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-4">
                <Text variant="h3" className="mb-2 text-zinc-800">
                  <a
                    href="https://tailwindcss.com/docs/padding"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    Padding Classes
                  </a>
                </Text>
                <Text variant="small" className="mb-2 text-zinc-600">
                  Used for internal spacing within elements (same scale as
                  margin)
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  p-4 → padding: 1rem (16px)
                </div>
              </div>
            </div>
          </div>

          <div>
            <Text variant="h2" className="mb-4 text-zinc-800">
              FAQ
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <Text variant="h3" className="mb-2 text-zinc-800">
                  🤔 Why use spacing tokens?
                </Text>
                <div className="space-y-3 text-zinc-600">
                  <Text variant="small">
                    Always use spacing classes instead of arbitrary values.
                    Reasons:
                  </Text>
                  <ul className="ml-4 list-inside list-disc space-y-1 text-sm">
                    <li>Ensures consistent spacing relationships</li>
                    <li>
                      Makes responsive design easier with consistent ratios
                    </li>
                    <li>Provides a harmonious visual rhythm</li>
                    <li>Easier to maintain and update globally</li>
                    <li>Prevents spacing inconsistencies across the app</li>
                  </ul>
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-4">
                <Text variant="h3" className="mb-2 text-zinc-800">
                  📏 How to choose spacing values?
                </Text>
                <div className="space-y-2 text-zinc-600">
                  <Text variant="small">
                    • <strong>1-2:</strong> Tight spacing, form elements
                  </Text>
                  <Text variant="small">
                    • <strong>3-4:</strong> Default component spacing
                  </Text>
                  <Text variant="small">
                    • <strong>6-8:</strong> Section spacing
                  </Text>
                  <Text variant="small">
                    • <strong>12+:</strong> Major layout divisions
                  </Text>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Common Spacing Examples */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-4 border-b border-gray-200 pb-2 text-zinc-500"
          >
            Common Spacing Values
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            These are the most frequently used spacing values in our design
            system.
          </Text>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {commonSpacing.map((space) => (
            <div
              key={space.name}
              className="rounded-lg border border-gray-200 p-4"
            >
              <div className="mb-3 flex items-center justify-between">
                <Text variant="body-medium" className="font-mono text-zinc-800">
                  {space.class}
                </Text>
                <div className="text-right">
                  <div className="text-sm font-medium text-zinc-800">
                    {space.rem}
                  </div>
                  <div className="text-xs text-zinc-500">{space.px}</div>
                </div>
              </div>
              <div className="mb-3">
                <div
                  className="border-l-4 border-blue-500 bg-blue-100"
                  style={{ marginLeft: space.rem }}
                >
                  <div className="h-4"></div>
                </div>
              </div>
              <Text variant="small" className="text-zinc-600">
                {space.usage}
              </Text>
            </div>
          ))}
        </div>

        <StoryCode
          code={`// Common spacing examples
<div className="m-1">Tight spacing (4px)</div>
<div className="m-4">Medium spacing (16px)</div>
<div className="m-8">Large spacing (32px)</div>
<div className="m-12">Section spacing (48px)</div>

// Padding uses the same scale
<div className="p-4">Internal spacing (16px)</div>`}
        />
      </div>

      {/* Complete Spacing Scale */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-4 border-b border-gray-200 pb-2 text-zinc-500"
          >
            Complete Spacing Scale
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            All available spacing values in our design system. Each value works
            for both margin and padding.
          </Text>
        </div>

        <div className="space-y-4">
          {spacingScale.map((space) => (
            <div
              key={space.name}
              className="flex items-center rounded-lg border border-gray-200 p-4"
            >
              <div className="flex w-32 flex-col">
                <Text variant="body-medium" className="font-mono text-zinc-800">
                  {space.name}
                </Text>
                <Text variant="small" className="font-mono text-zinc-500">
                  {space.class}
                </Text>
              </div>
              <div className="flex w-32 flex-col text-right">
                <Text variant="small" className="font-medium text-zinc-800">
                  {space.rem}
                </Text>
                <Text variant="small" className="text-zinc-500">
                  {space.px}
                </Text>
              </div>
              <div className="ml-8 flex-1">
                <div
                  className="h-4 border-l-4 border-blue-500 bg-blue-100"
                  style={{ marginLeft: space.value }}
                ></div>
              </div>
            </div>
          ))}
        </div>

        <StoryCode
          code={`// Spacing scale examples
<div className="m-0">No margin (0px)</div>
<div className="m-px">1px margin</div>
<div className="m-1">0.25rem margin (4px)</div>
<div className="m-2">0.5rem margin (8px)</div>
<div className="m-4">1rem margin (16px)</div>
<div className="m-8">2rem margin (32px)</div>
<div className="m-16">4rem margin (64px)</div>

// Directional spacing
<div className="mt-4">Margin top</div>
<div className="mr-4">Margin right</div>
<div className="mb-4">Margin bottom</div>
<div className="ml-4">Margin left</div>
<div className="mx-4">Margin horizontal</div>
<div className="my-4">Margin vertical</div>

// Padding uses identical values
<div className="p-4 pt-8 px-6">Mixed padding</div>`}
        />
      </div>

      {/* Responsive Spacing */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-4 border-b border-gray-200 pb-2 text-zinc-500"
          >
            Responsive Spacing
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            Use responsive prefixes to adjust spacing at different breakpoints.
          </Text>
        </div>

        <div className="rounded-lg border border-gray-200 p-6">
          <Text variant="h3" className="mb-4 text-zinc-800">
            Responsive Example
          </Text>
          <div className="mb-4 rounded bg-blue-50 p-4">
            <div className="m-2 rounded bg-blue-200 p-2 md:m-4 lg:m-8">
              <Text variant="small" className="text-blue-800">
                This element has m-2 on mobile, m-4 on tablet, and m-8 on
                desktop
              </Text>
            </div>
          </div>
          <StoryCode
            code={`<div className="m-2 md:m-4 lg:m-8">
  Responsive spacing
</div>

// Breakpoints:
// sm: 640px and up
// md: 768px and up  
// lg: 1024px and up
// xl: 1280px and up
// 2xl: 1536px and up`}
          />
        </div>
      </div>

      {/* Negative Spacing */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-4 border-b border-gray-200 pb-2 text-zinc-500"
          >
            Negative Spacing
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            Negative margins can be used to pull elements closer or create
            overlapping effects.
          </Text>
        </div>

        <div className="rounded-lg border border-gray-200 p-6">
          <Text variant="h3" className="mb-4 text-zinc-800">
            Negative Margin Example
          </Text>
          <div className="mb-4 space-y-4">
            <div className="rounded bg-gray-100 p-4">
              <div className="rounded bg-blue-200 p-4">First element</div>
              <div className="-mt-2 rounded bg-green-200 p-4">
                Second element with -mt-2 (pulls up by 8px)
              </div>
            </div>
          </div>
          <StoryCode
            code={`<div className="bg-blue-200 p-4">First element</div>
<div className="bg-green-200 p-4 -mt-2">
  Pulled up with negative margin
</div>

// Available negative values:
// -m-px, -m-0.5, -m-1, -m-1.5, -m-2, etc.
// Works with all directional classes: -mt-, -mr-, -mb-, -ml-, -mx-, -my-`}
          />
        </div>
      </div>
    </div>
  );
}

type Story = StoryObj<typeof meta>;
