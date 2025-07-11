import { Text } from "@/components/atoms/Text/Text";
import { colors } from "@/components/styles/colors";
import type { Meta } from "@storybook/nextjs";
import { StoryCode } from "./helpers/StoryCode";

const meta: Meta = {
  title: "Tokens /Colors",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

// Helper function to convert hex to RGB
function hexToRgb(hex: string): string {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return hex;

  const r = parseInt(result[1], 16);
  const g = parseInt(result[2], 16);
  const b = parseInt(result[3], 16);

  return `rgb(${r}, ${g}, ${b})`;
}

// Generate color categories from colors.ts
const colorCategories = Object.entries(colors)
  .filter(([key]) => !["white", "black"].includes(key))
  .map(([colorName, colorShades]) => {
    const descriptions: Record<string, string> = {
      slate: "Cool gray tones for modern, professional interfaces",
      zinc: "Neutral gray scale for backgrounds and subtle elements",
      red: "Error states, warnings, and destructive actions",
      orange: "Warnings, notifications, and secondary call-to-actions",
      yellow: "Highlights, cautions, and attention-grabbing elements",
      green: "Success states, confirmations, and positive actions",
      purple: "Brand accents, premium features, and creative elements",
      pink: "Highlights, special promotions, and playful interactions",
    };

    return {
      name: colorName.charAt(0).toUpperCase() + colorName.slice(1),
      description: descriptions[colorName] || `${colorName} color variations`,
      colors: Object.entries(colorShades as Record<string, string>).map(
        ([shade, hex]) => ({
          name: `${colorName}-${shade}`,
          hex,
          rgb: hexToRgb(hex),
          class: `bg-${colorName}-${shade}`,
          textClass: `text-${colorName}-${shade}`,
        }),
      ),
    };
  });

// Special colors from colors.ts
const specialColors = [
  {
    name: "Text",
    description: "Primary text colors for content and typography",
    colors: [
      {
        name: "text-white",
        hex: colors.white,
        rgb: hexToRgb(colors.white),
        class: "text-white",
        bgClass: "bg-white",
      },
      {
        name: "text-black",
        hex: colors.black,
        rgb: hexToRgb(colors.black),
        class: "text-black",
        bgClass: "bg-black",
      },
    ],
  },
  {
    name: "Background",
    description: "Standard background colors for layouts and surfaces",
    colors: [
      {
        name: "bg-white",
        hex: colors.white,
        rgb: hexToRgb(colors.white),
        class: "bg-white",
        textClass: "text-white",
      },
      {
        name: "bg-light-grey",
        hex: colors.lightGrey,
        rgb: hexToRgb(colors.lightGrey),
        class: "bg-light-grey",
        textClass: "text-light-grey",
      },
    ],
  },
];

export function AllVariants() {
  return (
    <div className="space-y-12">
      {/* Color System Documentation */}
      <div className="space-y-8">
        <div>
          <Text variant="h1" className="mb-4 text-zinc-800">
            Color Palette
          </Text>
          <Text variant="large" className="text-zinc-600">
            Use only these approved colors in your components. Many are named
            like Tailwind&apos;s default theme but override those values with
            our custom palette.
          </Text>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              How to Use
            </Text>
            <div className="space-y-4">
              <div className="rounded-lg border border-gray-200 p-4">
                <Text variant="body" className="mb-2 text-zinc-600">
                  Use any of the approved colors combined with{" "}
                  <a
                    href="https://tailwindcss.com/docs/colors"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500"
                  >
                    tailwind classes
                  </a>
                </Text>
                <div className="font-mono text-sm text-zinc-800">
                  bg-slate-500 → background-color: {colors.slate[500]}
                </div>
              </div>

              <div className="rounded-lg border-2 border-dashed border-red-200 bg-red-50 p-4">
                <Text
                  variant="body-medium"
                  className="mb-2 font-semibold text-red-800"
                >
                  ⚠️ Only Use These Colors
                </Text>
                <Text variant="body" className="text-red-700">
                  These are the ONLY approved colors. Don&apos;t use other
                  Tailwind colors or arbitrary values.
                </Text>
              </div>
            </div>
          </div>

          <div>
            <Text
              variant="h2"
              className="mb-2 text-xl font-semibold text-zinc-800"
            >
              Color Selection Guide
            </Text>
            <div className="space-y-2 text-zinc-600">
              <Text variant="body">
                • <strong>50-200:</strong> Light backgrounds, subtle borders
              </Text>
              <Text variant="body">
                • <strong>300-500:</strong> Interactive elements, primary colors
              </Text>
              <Text variant="body">
                • <strong>600-900:</strong> Text colors, dark backgrounds
              </Text>
              <Text variant="body">
                • <strong>Semantic:</strong> Red for errors, green for success
              </Text>
            </div>
          </div>
        </div>
      </div>

      {/* Special Colors */}
      <div className="space-y-8">
        <div className="grid gap-8 md:grid-cols-2">
          {specialColors.map((category) => (
            <div key={category.name} className="space-y-4">
              <div>
                <Text
                  variant="h3"
                  className="mb-1 text-lg font-semibold text-zinc-800"
                >
                  {category.name}
                </Text>
                <Text variant="body" className="text-zinc-600">
                  {category.description}
                </Text>
              </div>
              <div className="grid gap-3 sm:grid-cols-1">
                {category.colors.map((color) => (
                  <div
                    key={color.name}
                    className="flex items-center gap-4 rounded-lg border border-gray-200 p-4"
                  >
                    <div
                      className="h-12 w-12 flex-shrink-0 rounded border border-gray-300"
                      style={{ backgroundColor: color.hex }}
                    ></div>
                    <div className="flex-1 space-y-1">
                      <Text
                        variant="body-medium"
                        className="font-mono text-zinc-800"
                      >
                        {color.name}
                      </Text>
                      <Text variant="small" className="font-mono text-zinc-500">
                        {color.class}
                      </Text>
                      <div className="space-y-0.5">
                        <p className="font-mono text-xs text-zinc-500">
                          {color.hex}
                        </p>
                        <p className="font-mono text-xs text-zinc-500">
                          {color.rgb}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Color Categories */}
      <div className="space-y-12">
        {colorCategories.map((category) => (
          <div key={category.name} className="space-y-4">
            <div>
              <Text
                variant="h3"
                className="mb-1 text-lg font-semibold text-zinc-800"
              >
                {category.name}
              </Text>
              <Text variant="body" className="text-zinc-600">
                {category.description}
              </Text>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {category.colors.map((color) => (
                <div
                  key={color.name}
                  className="space-y-3 rounded-lg border border-gray-200 p-4"
                >
                  <div
                    className="h-16 w-full rounded border border-gray-300"
                    style={{ backgroundColor: color.hex }}
                  ></div>
                  <div className="space-y-1">
                    <Text
                      variant="body-medium"
                      className="font-mono text-zinc-800"
                    >
                      {color.name}
                    </Text>
                    <Text variant="small" className="font-mono text-zinc-500">
                      {color.class}
                    </Text>
                    <div className="space-y-0.5">
                      <p className="font-mono text-xs text-zinc-500">
                        {color.hex}
                      </p>
                      <p className="font-mono text-xs text-zinc-500">
                        {color.rgb}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Usage Examples */}
      <div className="space-y-8">
        <div>
          <Text
            variant="h2"
            className="mb-2 text-xl font-semibold text-zinc-800"
          >
            Usage Examples
          </Text>
        </div>

        <StoryCode
          code={`// ✅ CORRECT - Use approved design tokens
<div className="bg-slate-100 text-slate-800">
  Content with approved colors
</div>

<button className="bg-green-500 text-white hover:bg-green-600">
  Success Button
</button>

// Text colors
<h1 className="text-black">Primary heading</h1>
<p className="text-zinc-600">Secondary text</p>

// Semantic usage
<div className="bg-green-50 border-green-200 text-green-800">Success</div>
<div className="bg-red-50 border-red-200 text-red-800">Error</div>
<div className="bg-yellow-50 border-yellow-200 text-yellow-800">Warning</div>
<div className="bg-purple-50 border-purple-200 text-purple-800">Premium</div>
<div className="bg-pink-50 border-pink-200 text-pink-800">Special</div>

// ❌ INCORRECT - Don't use these  
<div className="bg-blue-500 text-purple-600">❌ Not approved</div>
<div className="bg-[#1234ff]">❌ Arbitrary values</div>`}
        />
      </div>
    </div>
  );
}
