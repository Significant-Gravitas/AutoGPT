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

// [class, value, class it replaces]. Each class is written out in full so
// Tailwind generates it.
type Token = [string, string, string];

const fontSizes: Token[] = [
  ["text-h1", "2.75rem", "text-[2.75rem]"],
  ["text-h2", "2rem", "text-[2rem]"],
  ["text-h3", "1.75rem", "text-[1.75rem]"],
  ["text-h4", "1.375rem", "text-[1.375rem]"],
  ["text-h5", "1rem", "text-[1rem]"],
  ["text-lead", "1.25rem", "text-[1.25rem]"],
  ["text-large", "1rem", "text-[1rem]"],
  ["text-body", "0.875rem", "text-[0.875rem]"],
  ["text-small", "0.75rem", "text-[0.75rem]"],
  ["text-label", "0.6785rem", "text-[0.6785rem]"],
  ["text-eyebrow", "0.75rem", "text-[0.75rem]"],
  ["text-5", "0.3125rem", "text-[5px]"],
  ["text-9", "0.5625rem", "text-[9px]"],
  ["text-10", "0.625rem", "text-[10px]"],
  ["text-11", "0.6875rem", "text-[11px]"],
  ["text-12", "0.75rem", "text-[12px]"],
  ["text-13", "0.8125rem", "text-[13px]"],
  ["text-14", "0.875rem", "text-[14px]"],
  ["text-15", "0.9375rem", "text-[15px]"],
  ["text-16", "1rem", "text-[16px]"],
  ["text-20", "1.25rem", "text-[20px]"],
  ["text-22", "1.375rem", "text-[22px]"],
  ["text-23", "1.4375rem", "text-[23px]"],
  ["text-26", "1.625rem", "text-[26px]"],
  ["text-28", "1.75rem", "text-[28px]"],
];

const lineHeights: Token[] = [
  ["leading-h1", "3.5rem", "leading-[3.5rem]"],
  ["leading-h2", "2.5rem", "leading-[2.5rem]"],
  ["leading-h3", "2.5rem", "leading-[2.5rem]"],
  ["leading-h4", "1.5rem", "leading-[1.5rem]"],
  ["leading-h5", "1.5rem", "leading-[1.5rem]"],
  ["leading-lead", "1.75rem", "leading-[1.75rem]"],
  ["leading-large", "1.625rem", "leading-[1.625rem]"],
  ["leading-body", "1.375rem", "leading-[1.375rem]"],
  ["leading-small", "1.125rem", "leading-[1.125rem]"],
  ["leading-label", "1.25rem", "leading-[1.25rem]"],
  ["leading-eyebrow", "1rem", "leading-[1rem]"],
  ["leading-4.5", "1.125rem", "leading-[18px]"],
  ["leading-5.5", "1.375rem", "leading-[22px]"],
  ["leading-6.5", "1.625rem", "leading-[26px]"],
  ["leading-14", "3.5rem", "leading-[56px]"],
];

const letterSpacings: Token[] = [
  ["tracking-h1", "-0.033rem", "tracking-[-0.033rem]"],
  ["tracking-h2", "-0.02rem", "tracking-[-0.02rem]"],
  ["tracking-h3", "-0.01313rem", "tracking-[-0.01313rem]"],
  ["tracking-label", "0.06785rem", "tracking-[0.06785rem]"],
  ["tracking-eyebrow", "0.06em", "tracking-[0.06em]"],
  ["tracking-display", "-0.04em", "tracking-[-0.04em]"],
  ["tracking-heading", "-0.02em", "tracking-[-0.02em]"],
  ["tracking-title", "-0.01em", "tracking-[-0.01em]"],
  ["tracking-caps", "0.12em", "tracking-[0.12em]"],
  ["tracking-caps-wide", "0.14em", "tracking-[0.14em]"],
];

// "1.375rem" -> "1.375rem · 22px" at a 16px root; em values stay as they are.
function describe(value: string) {
  if (!value.endsWith("rem")) return value;
  return `${value} · ${parseFloat(value) * 16}px`;
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
          Font sizes, line heights and letter spacing at the exact values in use
          today. A size never sets a line-height, so text-13 is a drop-in for
          text-[13px]. Named sizes are the Text variants; numbered sizes are px
          at a 16px root.
        </Text>
      </div>
      <Section
        title="Font size"
        description="font-size only"
        tokens={fontSizes}
        sampleClassName="font-sans text-zinc-900"
      />
      <Section
        title="Line height"
        description="Shown on a two-line sample at 14px"
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
