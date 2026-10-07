import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import type { Meta } from "@storybook/nextjs-vite";
import { StoryCode } from "./helpers/StoryCode";
import {
  formatPx,
  getSpacingScale,
  SPACING_UNIT,
  utilityClass,
} from "./helpers/theme";

const meta: Meta = {
  title: "Tokens /Spacing",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
    a11y: { test: "error" },
  },
};

export default meta;

const spacingScale = getSpacingScale();

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
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              Tailwind utilities
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-zinc-200 p-4">
                <a
                  href="https://tailwindcss.com/docs/margin"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mb-2 inline-flex flex-row items-center gap-1 text-base font-semibold text-blue-600 hover:underline"
                >
                  Margin Classes{" "}
                  <Icon icon={LinkSquare02Icon} size={12} aria-hidden />
                </a>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Used for external spacing between elements
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  m-4 → margin: 1rem (16px)
                </div>
              </div>
              <div className="rounded-lg border border-zinc-200 p-4">
                <a
                  href="https://tailwindcss.com/docs/padding"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mb-2 inline-flex flex-row items-center gap-1 text-base font-semibold text-blue-600 hover:underline"
                >
                  Padding Classes
                  <Icon icon={LinkSquare02Icon} size={12} aria-hidden />
                </a>

                <Text variant="body" className="mb-2 text-zinc-600">
                  Used for internal spacing within elements (same scale as
                  margin)
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  p-4 → padding: 1rem (16px)
                </div>
              </div>
              <Text variant="body" className="mb-4 text-zinc-600">
                We follow Tailwind CSS spacing system, which means you can use
                any spacing token available in the default Tailwind theme for
                margins and padding.
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
                🤔 Why use spacing tokens?
              </Text>
              <div className="space-y-3 text-zinc-600">
                <Text variant="body">
                  Always use spacing classes instead of arbitrary values.
                  Reasons:
                </Text>
                <ul className="ml-4 list-disc space-y-1 text-sm">
                  <li>Ensures consistent spacing relationships</li>
                  <li>Makes responsive design easier with consistent ratios</li>
                  <li>Provides a harmonious visual rhythm</li>
                  <li>Easier to maintain and update globally</li>
                  <li>Prevents spacing inconsistencies across the app</li>
                </ul>
              </div>
              <div>
                <Text
                  variant="h3"
                  className="mb-2 text-base font-semibold text-zinc-800"
                >
                  📏 How to choose spacing values?
                </Text>
                <div className="space-y-2 text-zinc-600">
                  <Text variant="body">
                    • <strong>1-2:</strong> Tight spacing, form elements
                  </Text>
                  <Text variant="body">
                    • <strong>3-4:</strong> Default component spacing
                  </Text>
                  <Text variant="body">
                    • <strong>6-8:</strong> Section spacing
                  </Text>
                  <Text variant="body">
                    • <strong>12+:</strong> Major layout divisions
                  </Text>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Complete Spacing Scale */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-2 text-xl font-semibold text-zinc-800"
          >
            Complete Spacing Scale
          </Text>
          <Text variant="body" className="mb-6 text-zinc-600">
            Tailwind derives every spacing step from one variable,{" "}
            <code>--spacing</code> ({SPACING_UNIT}): <code>p-4</code> is{" "}
            <code>calc(var(--spacing) * 4)</code>. Any multiple of 0.25 is a
            valid step, so <code>h-4.5</code> and <code>w-18</code> work without
            configuration; the rows below are the common steps. Each value works
            for margin, padding, gap, width and height.
          </Text>
        </div>

        <div className="space-y-4">
          {spacingScale.map((space) => (
            <div
              key={space.name}
              className="flex items-center rounded-lg border border-zinc-200 p-4"
            >
              <div className="flex w-32 flex-col">
                <Text variant="body-medium" className="font-mono text-zinc-800">
                  {space.name}
                </Text>
                <Text
                  variant="small"
                  className="font-mono text-muted-foreground"
                >
                  {utilityClass("m", space.name)}
                </Text>
              </div>
              <div className="flex w-32 flex-col text-right">
                <Text
                  variant="small"
                  className="font-mono text-muted-foreground"
                >
                  {space.value}
                </Text>
                <Text
                  variant="small"
                  className="font-mono text-muted-foreground"
                >
                  {formatPx(space.px)}
                </Text>
              </div>
              <div className="ml-8 flex-1">
                <div className="relative h-6 bg-zinc-50">
                  <div
                    className="absolute top-0 left-0 h-full bg-blue-500"
                    style={{ width: space.value }}
                  ></div>
                </div>
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
    </div>
  );
}
