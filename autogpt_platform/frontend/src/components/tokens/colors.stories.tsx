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
        name: "bg-bgLightGrey",
        hex: colors.bgLightGrey,
        rgb: hexToRgb(colors.bgLightGrey),
        class: "bg-bgLightGrey",
        textClass: "text-bgLightGrey",
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
          <Text variant="h1" className="mb-4 text-foreground">
            Color tokens
          </Text>
          <Text variant="large" className="text-muted-foreground">
            New UI uses role tokens from globals.css. The ramps below are the
            primitive layer. Feature code does not style with a ramp step.
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
              <div className="rounded-lg border border-border p-4">
                <Text variant="body" className="mb-2 text-muted-foreground">
                  Paint with a role. bg-background, text-foreground,
                  border-border, bg-primary, text-muted-foreground, and
                  bg-destructive are the API.
                </Text>
                <div className="font-mono text-sm text-foreground">
                  bg-primary → hsl(var(--primary))
                </div>
              </div>

              <div className="rounded-lg border-2 border-dashed border-red-200 bg-red-50 p-4">
                <Text
                  variant="body-medium"
                  className="mb-2 font-semibold text-red-800"
                >
                  Roles, then ramps
                </Text>
                <Text variant="body" className="text-red-700">
                  Reach for a role token first. A ramp step is a primitive.
                  Other palette families and arbitrary values are not the API.
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
                    className="flex items-center gap-4 rounded-lg border border-border p-4"
                  >
                    <div
                      className="h-12 w-12 flex-shrink-0 rounded border border-border"
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
                  className="space-y-3 rounded-lg border border-border p-4"
                >
                  <div
                    className="h-16 w-full rounded border border-border"
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

        <div className="max-w-3xl space-y-3 rounded-lg border border-border p-4">
          <Text variant="h3" className="text-foreground">
            Ramp defects, left unchanged
          </Text>
          <Text variant="body" className="text-muted-foreground">
            red 900 in colors.ts has a trailing space, so it is not a valid
            hex. slate 700 and slate 800 are both #475263. The slate ramp has
            no 950 step, so a slate 950 utility is still the default scale.
            customGray in the Tailwind config has no class use under src. The
            flat key bgLightGrey compiles to bg-bgLightGrey. These values stay
            as they are in this change.
          </Text>
        </div>

        <StoryCode
          code={`<div className="bg-background text-foreground">
  Page surface
</div>

<button className="bg-primary text-primary-foreground">
  Primary action
</button>

<p className="text-muted-foreground">Secondary text</p>
<p className="text-destructive">Something went wrong</p>`}
        />
      </div>
    </div>
  );
}
