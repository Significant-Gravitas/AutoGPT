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
        hex: colors.bgLightGrey,
        rgb: hexToRgb(colors.bgLightGrey),
        class: "bg-light-grey",
        textClass: "text-light-grey",
      },
    ],
  },
];

// Colour roles from src/app/globals.css. Swatches read the CSS variable
// directly, so the story always shows what the token renders.
const roleGroups = [
  {
    name: "Surfaces",
    roles: [
      ["canvas", "#F6F7F8", "body bg-[#F6F7F8]"],
      ["surface", "#FEFEFE", "bg-white"],
      ["surface-muted", "#F9F9FA", "bg-zinc-50"],
      ["surface-hover", "#EFEFF0", "bg-zinc-100"],
      ["surface-active", "#DADADC", "bg-zinc-200"],
      ["overlay", "#78716C", "bg-stone-500 (used at /20)"],
    ],
  },
  {
    name: "Text",
    roles: [
      ["foreground-base", "#141414", "text-black"],
      ["foreground-strong", "#2C2C30", "text-zinc-900"],
      ["foreground-emphasis", "#3E3E43", "text-zinc-800"],
      ["foreground-body", "#505057", "text-zinc-700"],
      ["foreground-secondary", "#68686F", "text-zinc-600"],
      ["foreground-tertiary", "#83838C", "text-zinc-500"],
      ["foreground-subtle", "#ADADB3", "text-zinc-400"],
      ["foreground-disabled", "#C5C5C9", "text-zinc-300"],
      ["placeholder", "#83838C", "placeholder:text-zinc-500"],
    ],
  },
  {
    name: "Borders",
    roles: [
      ["separator", "#EFEFF0", "border-zinc-100"],
      ["border-base", "#DADADC", "border-zinc-200"],
      ["border-strong", "#C5C5C9", "border-zinc-300"],
      ["focus", "#0A0A0A", "ring-neutral-950"],
    ],
  },
  {
    name: "Action",
    roles: [
      ["action", "#3E3E43", "bg-zinc-800"],
      ["action-hover", "#2C2C30", "bg-zinc-900"],
      ["action-foreground", "#FEFEFE", "text-white"],
    ],
  },
  {
    name: "Accent",
    roles: [
      ["brand", "#7733F5", "purple-500"],
      ["brand-hover", "#6C2EDF", "bg-purple-600"],
      ["accent-surface", "#F5F3FF", "bg-violet-50"],
      ["accent-strong", "#6D28D9", "text-violet-700"],
    ],
  },
  {
    name: "Status",
    roles: [
      ["error", "#EF4444", "red-500"],
      ["error-hover", "#D93636", "red-600"],
      ["error-foreground", "#AA3030", "text-red-700"],
      ["error-surface", "#FEF5F5", "bg-red-50"],
      ["warning", "#F59E0B", "amber-500"],
      ["warning-foreground", "#B45309", "text-amber-700"],
      ["warning-surface", "#FFFBEB", "bg-amber-50"],
      ["warning-border", "#FDE68A", "amber-200"],
      ["success", "#149443", "text-green-600"],
      ["success-foreground", "#107435", "text-green-700"],
      ["success-surface", "#E8F6ED", "bg-green-50"],
      ["info", "#3B82F6", "blue-500"],
      ["info-foreground", "#2563EB", "text-blue-600"],
      ["info-surface", "#EFF6FF", "bg-blue-50"],
    ],
  },
  {
    name: "shadcn names (values unchanged)",
    roles: [
      ["background", "#FAFAFA", "bg-background"],
      ["foreground", "#09090B", "text-foreground"],
      ["muted", "#F4F4F5", "bg-muted"],
      ["muted-foreground", "#71717A", "text-muted-foreground"],
      ["primary", "#18181B", "bg-primary"],
      ["accent", "#7C3BED", "bg-accent"],
      ["destructive", "#EF4444", "bg-destructive"],
      ["border", "#E4E4E7", "every border"],
      ["input", "#D6D6DB", "border-input"],
      ["ring", "#18181B", "ring-ring"],
    ],
  },
];

export function Roles() {
  return (
    <div className="space-y-10 p-8">
      <div className="space-y-2">
        <Text variant="h1">Colour roles</Text>
        <Text variant="large" tone="secondary">
          Name a colour by its job. Each role is a CSS variable in globals.css
          and a Tailwind colour, so it works with every utility and opacity
          modifier: bg-surface, text-foreground-tertiary, border-border-base,
          bg-overlay/20.
        </Text>
      </div>
      {roleGroups.map((group) => (
        <div key={group.name} className="space-y-4">
          <Text variant="h3">{group.name}</Text>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {group.roles.map(([name, hex, replaces]) => (
              <div
                key={name}
                className="flex items-center gap-4 rounded-lg border border-zinc-200 p-4"
              >
                <div
                  className="h-12 w-12 flex-shrink-0 rounded border border-zinc-300"
                  style={{ backgroundColor: `rgb(var(--color-${name}))` }}
                />
                <div className="min-w-0 space-y-1">
                  <Text variant="body-medium" className="font-mono">
                    {name}
                  </Text>
                  <Text variant="small" tone="muted" className="font-mono">
                    {hex} · replaces {replaces}
                  </Text>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

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
