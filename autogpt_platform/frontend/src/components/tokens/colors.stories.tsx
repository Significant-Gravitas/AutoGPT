import { Text } from "@/components/atoms/Text/Text";
import { COLOR_ROLE_GROUPS } from "@/components/styles/color-roles";
import type { Meta } from "@storybook/nextjs";

const meta: Meta = {
  title: "Tokens /Colors",
  parameters: {
    layout: "fullscreen",
    controls: { disable: true },
  },
};

export default meta;

// The `--color-*` declarations of every :root rule the page has loaded: the
// RGB triplet of each role, as written in src/app/tokens/theme.css.
function readRootColors() {
  const declared = new Map<string, string>();
  function visit(rules: CSSRuleList) {
    for (const rule of Array.from(rules)) {
      if (rule instanceof CSSStyleRule && rule.selectorText === ":root") {
        for (const property of Array.from(rule.style)) {
          if (property.startsWith("--color-"))
            declared.set(
              property.slice("--color-".length),
              rule.style.getPropertyValue(property).trim(),
            );
        }
      }
      if ("cssRules" in rule) visit((rule as CSSGroupingRule).cssRules);
    }
  }
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      visit(sheet.cssRules);
    } catch {
      // A cross-origin stylesheet (fonts) cannot be read; it holds no roles.
    }
  }
  return declared;
}

function toHex(triplet: string | undefined) {
  const channels = triplet?.split(/\s+/).map(Number);
  if (!channels || channels.length !== 3 || channels.some(Number.isNaN))
    return "not declared";
  return `#${channels
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")
    .toUpperCase()}`;
}

interface SwatchProps {
  name: string;
  detail: string;
}

function Swatch({ name, detail }: SwatchProps) {
  return (
    <div className="flex items-center gap-4 rounded-lg border border-border-base p-4">
      <div
        className="h-12 w-12 flex-shrink-0 rounded border border-border-strong"
        style={{ backgroundColor: `rgb(var(--color-${name}))` }}
      />
      <div className="min-w-0 space-y-1">
        <Text variant="body-medium" className="font-mono">
          {name}
        </Text>
        <Text variant="small" tone="muted" className="font-mono">
          {detail}
        </Text>
      </div>
    </div>
  );
}

export function Roles() {
  const declared = readRootColors();
  return (
    <div className="space-y-10 p-8">
      <div className="space-y-2">
        <Text variant="h1">Colour roles</Text>
        <Text variant="large" tone="secondary">
          Name a colour by its job. Each role is a CSS variable in
          tokens/theme.css and a Tailwind colour, so it works with every utility
          and opacity modifier: bg-surface, text-foreground-tertiary,
          border-border-base, bg-overlay/20. The values below are read from the
          stylesheet.
        </Text>
      </div>
      {COLOR_ROLE_GROUPS.map((group) => (
        <div key={group.name} className="space-y-4">
          <div className="space-y-1">
            <Text variant="h3">{group.name}</Text>
            <Text variant="body" tone="secondary">
              {group.description}
            </Text>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {group.roles.map((name) => (
              <Swatch
                key={name}
                name={name}
                detail={toHex(declared.get(name))}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
