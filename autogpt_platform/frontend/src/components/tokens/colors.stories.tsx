import { Text } from "@/components/atoms/Text/Text";
import type { Meta } from "@storybook/nextjs";
import { StoryCode } from "./helpers/StoryCode";
import {
  getPalette,
  getPaletteSingles,
  getSemanticColors,
} from "./helpers/theme";

const meta: Meta = {
  title: "Tokens /Colors",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
    a11y: { test: "error" },
  },
};

export default meta;

const FAMILY_ROLES: Record<string, string> = {
  zinc: "The neutral ramp: text, borders, fills and the primary action",
  slate: "A cool neutral, used in a few places",
  red: "Errors and destructive actions",
  orange: "Warning icons",
  yellow: "Warnings",
  green: "Success",
  purple: "Brand accent and focus rings",
  pink: "Decorative",
  blue: "Undecided: Tailwind's default, kept until there is a decision",
  sky: "Undecided: Tailwind's default, kept until there is a decision",
  teal: "Undecided: Tailwind's default, kept until there is a decision",
  cyan: "Undecided: Tailwind's default, kept until there is a decision",
};

const palette = getPalette();
const singles = getPaletteSingles();
const semanticColors = getSemanticColors();

interface SwatchProps {
  variable: string;
  label: string;
  detail: string;
  secondaryDetail?: string;
}

function Swatch({ variable, label, detail, secondaryDetail }: SwatchProps) {
  return (
    <div className="space-y-3 rounded-lg border border-border p-4">
      <div
        className="h-16 w-full rounded-sm border border-zinc-300"
        style={{ backgroundColor: `var(${variable})` }}
      />
      <div className="space-y-1">
        <Text variant="body-medium" className="font-mono">
          {label}
        </Text>
        <Text variant="small" tone="muted" className="font-mono">
          {detail}
        </Text>
        {secondaryDetail ? (
          <Text variant="small" tone="muted" className="font-mono">
            {secondaryDetail}
          </Text>
        ) : null}
      </div>
    </div>
  );
}

export function AllVariants() {
  return (
    <div className="space-y-12">
      <div className="space-y-4">
        <Text variant="h1">Colours</Text>
        <Text variant="large" tone="muted">
          Two layers, both read from <code>src/app/globals.css</code>. Semantic
          colours (<code>bg-background</code>,{" "}
          <code>text-muted-foreground</code>, <code>border-border</code>) say
          what a colour is for; use them where one fits. The palette below them
          is the only set of colour steps that compiles: Tailwind&apos;s own
          palette is switched off.
        </Text>
      </div>

      <section className="space-y-4">
        <Text variant="h3">Semantic</Text>
        <Text variant="body" tone="muted">
          Each class points at a palette step in <code>:root</code>; the{" "}
          <code>.dark</code> value is written but not active.
        </Text>
        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          {semanticColors.map((color) => (
            <Swatch
              key={color.name}
              variable={`--color-${color.name}`}
              label={color.name}
              detail={color.light}
              secondaryDetail={color.dark ? `dark: ${color.dark}` : undefined}
            />
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <Text variant="h3">White and black</Text>
        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
          {singles.map((color) => (
            <Swatch
              key={color.name}
              variable={`--color-${color.name}`}
              label={color.name}
              detail={color.value}
            />
          ))}
        </div>
      </section>

      {palette.map((family) => (
        <section key={family.name} className="space-y-4">
          <div className="space-y-1">
            <Text variant="h3" className="capitalize">
              {family.name}
            </Text>
            <Text variant="body" tone="muted">
              {FAMILY_ROLES[family.name] ?? ""}
            </Text>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
            {family.steps.map(({ step, value }) => (
              <Swatch
                key={step}
                variable={`--color-${family.name}-${step}`}
                label={`${family.name}-${step}`}
                detail={value}
              />
            ))}
          </div>
        </section>
      ))}

      <StoryCode
        code={`// Prefer semantic classes
<div className="bg-card text-card-foreground border-border" />
<Text tone="muted">Secondary copy</Text>          // text-muted-foreground
<div className="bg-success text-success-foreground" />

// Palette steps where no semantic class fits
<div className="bg-purple-50 text-purple-800" />

// Not available: Tailwind's palette and arbitrary colours
<div className="bg-gray-100" />    // does not compile
<div className="bg-[#1234ff]" />   // avoid`}
      />
    </div>
  );
}
