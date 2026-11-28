import { StoryCode } from "@/components/tokens/helpers/StoryCode";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { tones } from "@/components/styles/tones";
import { Text, textTones, textVariants, type TextProps } from "./Text";

const weights = ["normal", "medium", "semibold", "bold"] as const;
const leadings = [
  "none",
  "tight",
  "snug",
  "normal",
  "relaxed",
  "inherit",
  "4",
  "5",
  "6",
  "8",
] as const;

const meta: Meta<typeof Text> = {
  title: "Atoms/Text",
  component: Text,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
    controls: { hideNoControlsWarning: true },
    docs: {
      description: {
        component:
          "Renders text in a type role (`variant`: its size, family and the weight, tracking or line height the role always has). Props adjust the role: `weight`, `leading`, `tracking`, `mono`, `tabular`, `italic`, `decoration`, `transform`, `narrowSize`, `align` and `tone`. Uses Poppins for headings and Geist Sans for body text.",
      },
      source: {
        state: "open",
      },
    },
  },
  argTypes: {
    variant: {
      control: { type: "select" },
      options: textVariants,
      description: "Type role",
    },
    weight: {
      control: { type: "select" },
      options: [undefined, ...weights],
      description: "Font weight over the role's own",
    },
    leading: {
      control: { type: "select" },
      options: [undefined, ...leadings],
      description: "Line height over the role's own",
    },
    tracking: {
      control: { type: "select" },
      options: [undefined, "tight", "normal", "title", "caps", "eyebrow"],
      description: "Letter-spacing over the role's own",
    },
    narrowSize: {
      control: { type: "select" },
      options: [undefined, "body", "large"],
      description: "The size below the md breakpoint",
    },
    mono: { control: "boolean" },
    tabular: { control: "boolean" },
    italic: { control: "boolean" },
    decoration: {
      control: { type: "select" },
      options: [undefined, "underline", "line-through", "none"],
    },
    transform: {
      control: { type: "select" },
      options: [undefined, "uppercase", "lowercase", "capitalize", "none"],
    },
    tone: {
      control: { type: "select" },
      options: [undefined, ...textTones],
      description: "Semantic colour",
    },
    as: {
      control: { type: "select" },
      options: ["h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "div", "code"],
      description: "HTML element to render as",
    },
    children: {
      control: "text",
      description: "Text content",
    },
  },
};

export default meta;
type Story = StoryObj<typeof Text>;

function SectionTitle({ children }: { children: string }) {
  return (
    <div className="mb-4 border-b border-border-base pb-2">
      <Text variant="lead" tone="muted" as="h2">
        {children}
      </Text>
    </div>
  );
}

//=============================================================================
// Every role
//=============================================================================
export function Roles() {
  return (
    <div className="space-y-4 p-8">
      {textVariants.map((variant) => (
        <div key={variant} className="flex items-baseline gap-6">
          <Text variant="small" mono tone="muted" className="w-28 shrink-0">
            {variant}
          </Text>
          <Text variant={variant}>Find decision-makers at any company</Text>
        </div>
      ))}
      <StoryCode
        code={`<Text variant="h3">…</Text>
<Text variant="body">…</Text>
<Text variant="small" as="span">…</Text>`}
      />
    </div>
  );
}

//=============================================================================
// Roles × weight
//=============================================================================
export function Weights() {
  const roles = textVariants.filter(
    (variant) => variant !== "hero" && variant !== "h1",
  );
  return (
    <div className="space-y-3 p-8">
      <SectionTitle>Weight over the role&apos;s own</SectionTitle>
      {roles.map((variant) => (
        <div key={variant} className="flex items-baseline gap-6">
          <Text variant="small" mono tone="muted" className="w-28 shrink-0">
            {variant}
          </Text>
          {weights.map((weight) => (
            <Text key={weight} variant={variant} weight={weight} as="span">
              {weight}
            </Text>
          ))}
        </div>
      ))}
      <StoryCode
        code={`<Text variant="body" weight="medium">…</Text>
<Text variant="small" weight="semibold">…</Text>`}
      />
    </div>
  );
}

//=============================================================================
// Roles × leading. The parent's 30px line height shows what `inherit` keeps.
//=============================================================================
export function LineHeights() {
  const roles = ["large", "body", "body-sm", "small"] as const;
  return (
    <div className="space-y-6 p-8 leading-[30px]">
      <SectionTitle>Line height over the role&apos;s own</SectionTitle>
      {roles.map((variant) => (
        <div key={variant} className="space-y-2">
          <Text variant="small" mono tone="muted">
            {variant}
          </Text>
          <div className="grid grid-cols-5 gap-4">
            {[undefined, ...leadings].map((leading) => (
              <div key={leading ?? "role"} className="space-y-1">
                <Text variant="small" mono tone="subtle">
                  {leading ?? "(role)"}
                </Text>
                <div className="bg-surface-hover">
                  <Text variant={variant} leading={leading}>
                    Find decision-makers at any company, in two lines
                  </Text>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
      <StoryCode
        code={`<Text variant="small" leading="4">…</Text>
<Text variant="body" leading="5">…</Text>
<Text variant="body-sm" leading="inherit">…</Text>`}
      />
    </div>
  );
}

//=============================================================================
// Interactive Playground
//=============================================================================
export const Playground: Story = {
  args: {
    variant: "body",
    children:
      "Edit this text and try different roles and props via the controls below",
  },
  parameters: {
    controls: {
      include: [
        "variant",
        "weight",
        "leading",
        "tracking",
        "narrowSize",
        "mono",
        "tabular",
        "italic",
        "decoration",
        "transform",
        "tone",
        "as",
        "children",
      ],
    },
  },
  render: (args) => (
    <div className="space-y-8 p-8">
      <Text {...args} />
    </div>
  ),
};

//=============================================================================
// Custom Element
//=============================================================================
export function CustomElement() {
  return (
    <div className="space-y-8">
      <Text variant="h3" as="div">
        H3 size rendered as div
      </Text>
      <Text variant="body" as="h2">
        Body size rendered as h2
      </Text>
    </div>
  );
}

//=============================================================================
// Tones
//=============================================================================
export function Tones() {
  return (
    <div className="space-y-6 p-8">
      <Text variant="body">Default (black)</Text>
      <Text variant="body" tone="primary">
        Primary · zinc-900 · titles and values
      </Text>
      <Text variant="body" tone="secondary">
        Secondary · zinc-600 · descriptions
      </Text>
      <Text variant="body" tone="muted">
        Muted · zinc-500 · meta, captions, empty states
      </Text>
      <Text variant="body" tone="danger">
        Danger · red-600 · destructive copy
      </Text>
      <StoryCode
        code={`<Text variant="body" tone="primary">Primary</Text>
<Text variant="body" tone="secondary">Secondary</Text>
<Text variant="body" tone="muted">Muted</Text>
<Text variant="body" tone="danger">Danger</Text>`}
      />
    </div>
  );
}

//=============================================================================
// All tones, each on every role
//=============================================================================
export function AllTones() {
  return (
    <div className="space-y-8 p-8">
      <Text variant="body">Default (foreground-base)</Text>
      {textTones.map((tone) => (
        <div key={tone} className="space-y-1">
          <Text variant="small" weight="medium" tone="muted">
            {tone} · {tones[tone]}
          </Text>
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
            {textVariants.map((variant) => (
              <Text key={variant} variant={variant} tone={tone}>
                {variant}
              </Text>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

//=============================================================================
// Text roles (Home / Team)
//=============================================================================
export function TextRoles() {
  return (
    <div className="max-w-md space-y-3 p-8">
      <Text variant="eyebrow">Needs you</Text>
      <Text variant="large" weight="medium" tone="primary">
        Section title · large · medium
      </Text>
      <Text variant="body" weight="medium" tone="primary">
        Row title · body · medium
      </Text>
      <Text variant="body" tone="secondary">
        Description · body · secondary
      </Text>
      <Text variant="small" tone="muted" tabular>
        Meta · small · muted · 2 files · 1 action
      </Text>
      <Text variant="small" weight="medium" tone="muted">
        Chip label · small · medium
      </Text>
      <StoryCode
        code={`<Text variant="eyebrow">Needs you</Text>
<Text variant="large" weight="medium" tone="primary">Section title</Text>
<Text variant="body" weight="medium" tone="primary">Row title</Text>
<Text variant="body" tone="secondary">Description</Text>
<Text variant="small" tone="muted" tabular>Meta</Text>`}
      />
      <Text variant="small" tone="muted">
        Floor is 12px (small). No arbitrary px sizes.
      </Text>
    </div>
  );
}

//=============================================================================
// Modifier props: adjust a role without a className
//=============================================================================
const MODIFIERS: { label: string; props: Partial<TextProps> }[] = [
  { label: 'weight="semibold"', props: { weight: "semibold" } },
  { label: 'leading="5"', props: { leading: "5" } },
  { label: 'tracking="tight"', props: { tracking: "tight" } },
  { label: "mono · run_7f3a2c", props: { mono: true } },
  { label: "tabular · 1,204.50 credits", props: { tabular: true } },
  { label: "italic", props: { italic: true } },
  {
    label: 'decoration="line-through"',
    props: { decoration: "line-through", tone: "muted" },
  },
  { label: 'decoration="underline"', props: { decoration: "underline" } },
  {
    label: 'narrowSize="large" (resize below md)',
    props: { narrowSize: "large" },
  },
  { label: 'align="center"', props: { align: "center" } },
];

export function Modifiers() {
  return (
    <div className="space-y-3 p-8">
      {MODIFIERS.map(({ label, props }) => (
        <Text key={label} variant="body" {...props}>
          {label}
        </Text>
      ))}
      <Text variant="small" transform="uppercase" tone="muted">
        transform=&quot;uppercase&quot;
      </Text>
      <Text variant="small" transform="uppercase" tracking="caps">
        tracking=&quot;caps&quot;
      </Text>
      <StoryCode
        code={`<Text variant="body" weight="semibold">…</Text>
<Text variant="body" leading="5">…</Text>
<Text variant="small" mono leading="4">…</Text>
<Text variant="body" decoration="line-through" tone="muted">…</Text>
<Text variant="small" weight="medium" leading="4" narrowSize="body">…</Text>`}
      />
    </div>
  );
}
