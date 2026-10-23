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

const fontSizes: Token[] = [
  ["text-tiny", "0.625rem / 0.875rem", "text-[10px]"],
  ["text-caption-sm", "0.6875rem / 1rem", "text-[11px]"],
  ["text-caption", "0.75rem / 1.125rem", "Text small, text-[12px]"],
  ["text-body-sm", "0.8125rem / 1.25rem", "text-[13px]"],
  ["text-body", "0.875rem / 1.375rem", "Text body, text-[14px]"],
  ["text-body-md", "0.9375rem / 1.5rem", "text-[15px]"],
  ["text-body-lg", "1rem / 1.625rem", "Text large, text-[16px]"],
  ["text-lead", "1.25rem / 1.75rem", "Text lead"],
  ["text-heading-sm", "1.375rem / 1.5rem", "Text h4"],
  ["text-heading", "1.75rem / 2.5rem", "Text h3"],
  ["text-title", "2rem / 2.5rem", "Text h2"],
  ["text-display", "2.75rem / 3.5rem", "Text h1"],
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
      <div className="divide-y divide-zinc-200 rounded-lg border border-zinc-200">
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
        sampleClassName="font-sans text-zinc-900"
      />
      <Section
        title="Line height"
        description="Overrides, shown on a two-line sample at 14px"
        tokens={lineHeights}
        sampleClassName="max-w-[8rem] bg-zinc-50 font-sans text-sm text-zinc-900"
      />
      <Section
        title="Letter spacing"
        description="Shown at 16px; em values scale with the font size"
        tokens={letterSpacings}
        sampleClassName="font-sans text-base uppercase text-zinc-900"
      />
    </div>
  );
}
