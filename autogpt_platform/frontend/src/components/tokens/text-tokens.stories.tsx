import { Text } from "@/components/atoms/Text/Text";
import type { Meta } from "@storybook/nextjs";

const meta: Meta = {
  title: "Tokens /Text tokens",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

// [class, value, classes it replaces]. Each class is written out in full so
// Tailwind generates it.
type Token = [string, string, string];

// The type scale, the app's only text sizes (src/app/tokens/typography.css).
// The last column names the Tailwind sizes each step replaced.
const fontSizes: Token[] = [
  ["text-tiny", "0.625rem / 0.875rem", "10px"],
  ["text-caption-sm", "0.6875rem / 1rem", "11px"],
  ["text-caption", "0.75rem / 1.125rem", "Text small; 12px, Tailwind xs"],
  ["text-body-sm", "0.8125rem / 1.25rem", "13px"],
  ["text-body", "0.875rem / 1.375rem", "Text body; 14px, Tailwind sm"],
  ["text-body-md", "0.9375rem / 1.5rem", "15px"],
  ["text-body-lg", "1rem / 1.625rem", "Text large; 16px, Tailwind base"],
  ["text-lead", "1.25rem / 1.75rem", "Text lead; Tailwind lg and xl"],
  ["text-heading-sm", "1.375rem / 1.5rem", "Text h4"],
  ["text-heading-md", "1.5rem / 2rem", "Text heading-md; Tailwind 2xl"],
  ["text-heading", "1.75rem / 2.5rem", "Text h3; Tailwind 3xl"],
  ["text-title", "2rem / 2.5rem", "Text h2; Tailwind 4xl"],
  ["text-display", "2.75rem / 3.5rem", "Text h1; Tailwind 5xl"],
  ["text-hero", "4rem / 1", "Text hero; Tailwind 6xl to 9xl"],
];

const lineHeights: Token[] = [
  ["leading-inherit", "inherit", "a raw size that sets no line-height"],
  ["leading-4.5", "1.125rem", "leading-[18px]"],
  ["leading-5.5", "1.375rem", "leading-[22px]"],
  ["leading-6.5", "1.625rem", "leading-[26px]"],
];

const letterSpacings: Token[] = [
  ["tracking-display", "-0.04em", "tracking-[-0.04em]"],
  ["tracking-heading", "-0.02em", "tracking-[-0.02em]"],
  ["tracking-title", "-0.01em", "tracking-[-0.01em]"],
  ["tracking-caps", "0.12em", "tracking-[0.12em]"],
  ["tracking-eyebrow", "0.06em", "tracking-[0.06em]"],
];

// "0.8125rem / 1.25rem" -> "13px / 20px" at a 16px root.
function describe(value: string) {
  return value
    .split(" / ")
    .map((part) => (part.endsWith("rem") ? `${parseFloat(part) * 16}px` : part))
    .join(" / ");
}

interface SectionProps {
  title: string;
  description: string;
  tokens: Token[];
  sampleClassName: string;
}

function Section({
  title,
  description,
  tokens,
  sampleClassName,
}: SectionProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <Text variant="h3">{title}</Text>
        <Text variant="body" tone="secondary">
          {description}
        </Text>
      </div>
      <div className="divide-y divide-border-base rounded-lg border border-border-base">
        {tokens.map(([className, value, replaces]) => (
          <div
            key={className}
            className="grid grid-cols-1 gap-1 px-4 py-3 md:grid-cols-[12rem_14rem_1fr] md:items-center md:gap-4"
          >
            <Text variant="body-medium" className="font-mono">
              {className}
            </Text>
            <Text variant="small" tone="muted" className="font-mono">
              {describe(value)} · {replaces}
            </Text>
            <div className={`${sampleClassName} ${className}`}>
              The quick brown fox
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function AllTokens() {
  return (
    <div className="space-y-10 p-4 md:p-8">
      <div className="space-y-2">
        <Text variant="h1">Text tokens</Text>
        <Text variant="large" tone="secondary">
          One size per role, each with its line-height, at the values in use
          today. A raw size that sets no line-height becomes the role plus
          leading-inherit, so nothing moves.
        </Text>
      </div>
      <Section
        title="Size roles"
        description="font-size / bundled line-height"
        tokens={fontSizes}
        sampleClassName="font-sans text-foreground-strong"
      />
      <Section
        title="Line height"
        description="Overrides, shown on a two-line sample at 14px"
        tokens={lineHeights}
        sampleClassName="max-w-[8rem] bg-surface-muted font-sans text-body text-foreground-strong"
      />
      <Section
        title="Letter spacing"
        description="Shown at 16px; em values scale with the font size"
        tokens={letterSpacings}
        sampleClassName="font-sans text-body-lg uppercase text-foreground-strong"
      />
    </div>
  );
}
