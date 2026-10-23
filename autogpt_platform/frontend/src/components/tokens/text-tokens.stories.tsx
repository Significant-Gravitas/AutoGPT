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

// Each class is written out in full so Tailwind generates it.
const fontSizes = [
  { className: "text-h1", value: "2.75rem · 44px", replaces: "text-[2.75rem]" },
  { className: "text-h2", value: "2rem · 32px", replaces: "text-[2rem]" },
  { className: "text-h3", value: "1.75rem · 28px", replaces: "text-[1.75rem]" },
  {
    className: "text-h4",
    value: "1.375rem · 22px",
    replaces: "text-[1.375rem]",
  },
  { className: "text-h5", value: "1rem · 16px", replaces: "text-[1rem]" },
  {
    className: "text-lead",
    value: "1.25rem · 20px",
    replaces: "text-[1.25rem]",
  },
  { className: "text-large", value: "1rem · 16px", replaces: "text-[1rem]" },
  {
    className: "text-body",
    value: "0.875rem · 14px",
    replaces: "text-[0.875rem]",
  },
  {
    className: "text-small",
    value: "0.75rem · 12px",
    replaces: "text-[0.75rem]",
  },
  {
    className: "text-label",
    value: "0.6785rem · 10.856px",
    replaces: "text-[0.6785rem]",
  },
  {
    className: "text-eyebrow",
    value: "0.75rem · 12px",
    replaces: "text-[0.75rem]",
  },
  { className: "text-5", value: "0.3125rem · 5px", replaces: "text-[5px]" },
  { className: "text-9", value: "0.5625rem · 9px", replaces: "text-[9px]" },
  { className: "text-10", value: "0.625rem · 10px", replaces: "text-[10px]" },
  { className: "text-11", value: "0.6875rem · 11px", replaces: "text-[11px]" },
  { className: "text-12", value: "0.75rem · 12px", replaces: "text-[12px]" },
  { className: "text-13", value: "0.8125rem · 13px", replaces: "text-[13px]" },
  { className: "text-14", value: "0.875rem · 14px", replaces: "text-[14px]" },
  { className: "text-15", value: "0.9375rem · 15px", replaces: "text-[15px]" },
  { className: "text-16", value: "1rem · 16px", replaces: "text-[16px]" },
  { className: "text-20", value: "1.25rem · 20px", replaces: "text-[20px]" },
  { className: "text-22", value: "1.375rem · 22px", replaces: "text-[22px]" },
  { className: "text-23", value: "1.4375rem · 23px", replaces: "text-[23px]" },
  { className: "text-26", value: "1.625rem · 26px", replaces: "text-[26px]" },
  { className: "text-28", value: "1.75rem · 28px", replaces: "text-[28px]" },
];

const lineHeights = [
  {
    className: "leading-h1",
    value: "3.5rem · 56px",
    replaces: "leading-[3.5rem]",
  },
  {
    className: "leading-h2",
    value: "2.5rem · 40px",
    replaces: "leading-[2.5rem]",
  },
  {
    className: "leading-h3",
    value: "2.5rem · 40px",
    replaces: "leading-[2.5rem]",
  },
  {
    className: "leading-h4",
    value: "1.5rem · 24px",
    replaces: "leading-[1.5rem]",
  },
  {
    className: "leading-h5",
    value: "1.5rem · 24px",
    replaces: "leading-[1.5rem]",
  },
  {
    className: "leading-lead",
    value: "1.75rem · 28px",
    replaces: "leading-[1.75rem]",
  },
  {
    className: "leading-large",
    value: "1.625rem · 26px",
    replaces: "leading-[1.625rem]",
  },
  {
    className: "leading-body",
    value: "1.375rem · 22px",
    replaces: "leading-[1.375rem]",
  },
  {
    className: "leading-small",
    value: "1.125rem · 18px",
    replaces: "leading-[1.125rem]",
  },
  {
    className: "leading-label",
    value: "1.25rem · 20px",
    replaces: "leading-[1.25rem]",
  },
  {
    className: "leading-eyebrow",
    value: "1rem · 16px",
    replaces: "leading-[1rem]",
  },
  {
    className: "leading-4.5",
    value: "1.125rem · 18px",
    replaces: "leading-[18px]",
  },
  {
    className: "leading-5.5",
    value: "1.375rem · 22px",
    replaces: "leading-[22px]",
  },
  {
    className: "leading-6.5",
    value: "1.625rem · 26px",
    replaces: "leading-[26px]",
  },
  {
    className: "leading-14",
    value: "3.5rem · 56px",
    replaces: "leading-[56px]",
  },
];

const letterSpacings = [
  {
    className: "tracking-h1",
    value: "-0.033rem",
    replaces: "tracking-[-0.033rem]",
  },
  {
    className: "tracking-h2",
    value: "-0.02rem",
    replaces: "tracking-[-0.02rem]",
  },
  {
    className: "tracking-h3",
    value: "-0.01313rem",
    replaces: "tracking-[-0.01313rem]",
  },
  {
    className: "tracking-label",
    value: "0.06785rem",
    replaces: "tracking-[0.06785rem]",
  },
  {
    className: "tracking-eyebrow",
    value: "0.06em",
    replaces: "tracking-[0.06em]",
  },
  {
    className: "tracking-display",
    value: "-0.04em",
    replaces: "tracking-[-0.04em]",
  },
  {
    className: "tracking-heading",
    value: "-0.02em",
    replaces: "tracking-[-0.02em]",
  },
  {
    className: "tracking-title",
    value: "-0.01em",
    replaces: "tracking-[-0.01em]",
  },
  {
    className: "tracking-caps",
    value: "0.12em",
    replaces: "tracking-[0.12em]",
  },
  {
    className: "tracking-caps-wide",
    value: "0.14em",
    replaces: "tracking-[0.14em]",
  },
];

interface Token {
  className: string;
  value: string;
  replaces: string;
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
        {tokens.map((token) => (
          <div
            key={token.className}
            className="grid grid-cols-[12rem_14rem_1fr] items-center gap-4 px-4 py-3"
          >
            <Text variant="body-medium" className="font-mono">
              {token.className}
            </Text>
            <Text variant="small" tone="muted" className="font-mono">
              {token.value} · {token.replaces}
            </Text>
            <div className={`${sampleClassName} ${token.className}`}>
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
    <div className="space-y-10 p-8">
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
