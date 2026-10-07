import {
  tones,
  variantElementMap,
  variants,
  type Tone,
  type Variant,
} from "@/components/atoms/Text/helpers";
import { Text } from "@/components/atoms/Text/Text";
import type { Meta } from "@storybook/nextjs";
import { StoryCode } from "./helpers/StoryCode";

const meta: Meta<typeof Text> = {
  title: "Tokens /Typography",
  component: Text,
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
    a11y: { test: "error" },
  },
};

export default meta;

const variantNames = Object.keys(variants) as Variant[];
const toneNames = Object.keys(tones) as Tone[];

function isHeading(variant: Variant) {
  return variants[variant].includes("font-poppins");
}

const headingVariants = variantNames.filter(isHeading);
const bodyVariants = variantNames.filter((variant) => !isHeading(variant));

interface VariantRowProps {
  variant: Variant;
}

function VariantRow({ variant }: VariantRowProps) {
  return (
    <div className="flex flex-col gap-4 border-b border-zinc-200 py-6 md:flex-row">
      <div className="space-y-1 md:w-48 md:shrink-0">
        <Text variant="body-medium" className="font-mono text-zinc-800">
          {variant}
        </Text>
        <Text variant="small" className="font-mono text-zinc-500">
          &lt;{variantElementMap[variant]}&gt;
        </Text>
      </div>
      <div className="space-y-2">
        <Text variant={variant} as="div">
          The quick brown fox jumps over the lazy dog
        </Text>
        <Text variant="small" className="font-mono text-zinc-500">
          {variants[variant]}
        </Text>
      </div>
    </div>
  );
}

export function AllVariants() {
  return (
    <div className="space-y-12">
      {/* Typography System Documentation */}
      <div className="space-y-8">
        <div>
          <Text variant="h1" className="mb-4 text-zinc-800">
            Typography System
          </Text>
          <Text variant="large" className="text-zinc-600">
            Our typography system uses two carefully selected fonts to create a
            clear hierarchy and excellent readability across all interfaces.
          </Text>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <Text variant="h4" as="h2" className="mb-4 text-zinc-800">
              Font Families
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-zinc-200 p-4">
                <Text
                  variant="large-semibold"
                  as="h3"
                  className="mb-2 text-zinc-800"
                >
                  <a
                    href="https://fonts.google.com/specimen/Poppins"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    Poppins
                  </a>
                </Text>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Used for all headings and display text
                </Text>
                <div className="font-poppins text-2xl text-zinc-800">
                  The quick brown fox
                </div>
              </div>
              <div className="rounded-lg border border-zinc-200 p-4">
                <Text
                  variant="large-semibold"
                  as="h3"
                  className="mb-2 text-zinc-800"
                >
                  <a
                    href="https://github.com/vercel/geist-font"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    Geist Sans
                  </a>
                </Text>
                <Text variant="body" className="mb-2 text-zinc-600">
                  Used for all body text, labels, and UI elements
                </Text>
                <div className="font-sans text-base text-zinc-800">
                  The quick brown fox jumps over the lazy dog
                </div>
              </div>
            </div>
          </div>

          <div>
            <Text variant="h4" as="h2" className="mb-4 text-zinc-800">
              FAQ
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-zinc-200 p-4">
                <Text
                  variant="large-semibold"
                  as="h3"
                  className="mb-2 text-zinc-800"
                >
                  🤔 Why can&apos;t I use &lt;p&gt; tags directly?
                </Text>
                <div className="space-y-3 text-zinc-600">
                  <Text variant="body" className="text-zinc-600">
                    Always use the{" "}
                    <code className="rounded-sm bg-zinc-100 px-2 py-1 text-xs">
                      &lt;Text /&gt;
                    </code>{" "}
                    component instead of plain HTML elements like{" "}
                    <code className="rounded-sm bg-zinc-100 px-2 py-1 text-xs">
                      &lt;h1&gt;
                    </code>
                    ,{" "}
                    <code className="rounded-sm bg-zinc-100 px-2 py-1 text-xs">
                      &lt;p&gt;
                    </code>
                    ,{" "}
                    <code className="rounded-sm bg-zinc-100 px-2 py-1 text-xs">
                      &lt;span&gt;
                    </code>
                    , etc... Reasons:
                  </Text>
                  <ul className="ml-4 list-inside list-disc space-y-1 text-sm">
                    <li>Ensures consistent typography across the entire app</li>
                    <li>
                      Makes future design updates easier (change once, update
                      everywhere)
                    </li>
                    <li>Provides TypeScript safety for typography variants</li>
                    <li>
                      Automatically maps to correct HTML elements for
                      accessibility
                    </li>
                    <li>Prevents styling inconsistencies and design drift</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Typography Variants, read from atoms/Text/helpers.ts */}
      <div className="space-y-12">
        <div>
          <Text
            variant="h4"
            as="h2"
            className="border-b border-zinc-200 pb-2 text-zinc-500"
          >
            Headings (Poppins)
          </Text>
          {headingVariants.map((variant) => (
            <VariantRow key={variant} variant={variant} />
          ))}
        </div>

        <div>
          <Text
            variant="h4"
            as="h2"
            className="border-b border-zinc-200 pb-2 text-zinc-500"
          >
            Body and Labels (Geist Sans)
          </Text>
          {bodyVariants.map((variant) => (
            <VariantRow key={variant} variant={variant} />
          ))}
        </div>

        <div className="space-y-4">
          <Text
            variant="h4"
            as="h2"
            className="border-b border-zinc-200 pb-2 text-zinc-500"
          >
            Tones
          </Text>
          <div className="flex flex-wrap gap-8">
            {toneNames.map((tone) => (
              <div key={tone} className="space-y-1">
                <Text variant="body" tone={tone}>
                  tone=&quot;{tone}&quot;
                </Text>
                <Text variant="small" className="font-mono text-zinc-500">
                  {tones[tone]}
                </Text>
              </div>
            ))}
          </div>
        </div>

        <StoryCode
          code={variantNames
            .map((variant) => `<Text variant="${variant}">${variant}</Text>`)
            .join("\n")}
        />
      </div>
    </div>
  );
}
