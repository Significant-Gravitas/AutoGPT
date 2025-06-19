import { Text } from "@/components/atoms/Text/Text";
import type { Meta } from "@storybook/nextjs";
import { StoryCode } from "./helpers/StoryCode";

const meta: Meta<typeof Text> = {
  title: "Tokens /Typography",
  component: Text,
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

export function AllVariants() {
  return (
    <div className="space-y-12">
      {/* Typography System Documentation */}
      <div className="space-y-8">
        <div>
          <h1 className="mb-4 text-4xl font-bold text-zinc-800">
            Typography System
          </h1>
          <p className="text-lg leading-relaxed text-zinc-600">
            Our typography system uses two carefully selected fonts to create a
            clear hierarchy and excellent readability across all interfaces.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-800">
              Font Families
            </h2>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <h3 className="mb-2 font-semibold text-zinc-800">
                  <a
                    href="https://fonts.google.com/specimen/Poppins"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    Poppins
                  </a>
                </h3>
                <p className="mb-2 text-sm text-zinc-600">
                  Used for all headings and display text
                </p>
                <div className="font-poppins text-2xl text-zinc-800">
                  The quick brown fox
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-4">
                <h3 className="mb-2 font-semibold text-zinc-800">
                  <a
                    href="https://github.com/vercel/geist-font"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    Geist Sans
                  </a>
                </h3>
                <p className="mb-2 text-sm text-zinc-600">
                  Used for all body text, labels, and UI elements
                </p>
                <div className="font-sans text-base text-zinc-800">
                  The quick brown fox jumps over the lazy dog
                </div>
              </div>
            </div>
          </div>

          <div>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-800">FAQ</h2>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <h3 className="mb-2 font-semibold text-zinc-800">
                  🤔 Why can&apos;t I use &lt;p&gt; tags directly?
                </h3>
                <div className="space-y-3 text-zinc-600">
                  <p className="text-sm">
                    Always use the{" "}
                    <code className="rounded bg-gray-100 px-2 py-1 text-xs">
                      &lt;Text /&gt;
                    </code>{" "}
                    component instead of plain HTML elements like{" "}
                    <code className="rounded bg-gray-100 px-2 py-1 text-xs">
                      &lt;h1&gt;
                    </code>
                    ,{" "}
                    <code className="rounded bg-gray-100 px-2 py-1 text-xs">
                      &lt;p&gt;
                    </code>
                    ,{" "}
                    <code className="rounded bg-gray-100 px-2 py-1 text-xs">
                      &lt;span&gt;
                    </code>
                    , etc... Reasons:
                  </p>
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

      {/* Typography Examples */}
      <div className="space-y-8">
        <div className="mb-19 mb-20 space-y-6">
          <h2 className="mb-4 border-b border-border pb-2 text-xl text-zinc-500">
            Headings (Poppins)
          </h2>
          <Text variant="h1">Heading 1</Text>
          <Text variant="h2">Heading 2</Text>
          <Text variant="h3">Heading 3</Text>
          <Text variant="h4">Heading 4</Text>
          <StoryCode
            code={`<Text variant="h1">Heading 1</Text>
<Text variant="h2">Heading 2</Text>
<Text variant="h3">Heading 3</Text>
<Text variant="h4">Heading 4</Text>`}
          />
        </div>

        <h2 className="mb-4 border-b border-border pb-2 text-xl text-zinc-500">
          Body Text (Geist Sans)
        </h2>
        <Text variant="lead">Lead</Text>
        <StoryCode code="<Text variant='lead'>Lead</Text>" />
        <div className="flex flex-row gap-8">
          <Text variant="large">Large</Text>
          <Text variant="large-medium">Large Medium</Text>
          <Text variant="large-semibold">Large Semibold</Text>
        </div>
        <StoryCode
          code={`<Text variant="large">Large</Text>
<Text variant="large-medium">Large Medium</Text>
<Text variant="large-semibold">Large Semibold</Text>`}
        />
        <div className="flex flex-row gap-8">
          <Text variant="body">Body</Text>
          <Text variant="body-medium">Body Medium</Text>
        </div>
        <StoryCode
          code={`<Text variant="body">Body</Text>
<Text variant="body-medium">Body Medium</Text>`}
        />
        <div className="flex flex-row gap-8">
          <Text variant="small">Small</Text>
          <Text variant="small-medium">Small Medium</Text>
        </div>
        <StoryCode
          code={`<Text variant="small">Small</Text>
<Text variant="small-medium">Small Medium</Text>`}
        />
        <Text variant="subtle">Subtle</Text>
        <StoryCode code={`<Text variant="subtle">Subtle</Text>`} />
      </div>
    </div>
  );
}
