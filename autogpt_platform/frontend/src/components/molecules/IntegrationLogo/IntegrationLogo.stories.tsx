import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Text } from "@/components/atoms/Text/Text";
import { IntegrationLogo } from "./IntegrationLogo";

const SIZES = [16, 20, 24, 32, 48];

const PROVIDERS = [
  "github",
  "google",
  "notion",
  "linear",
  "openai",
  "anthropic",
  "discord",
  "hubspot",
];

const meta = {
  title: "Molecules/IntegrationLogo",
  component: IntegrationLogo,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "A provider's logo from `public/integrations`, keyed by a scrubbed provider slug. A provider without a logo file (or a slug that scrubs to nothing) degrades to a neutral puzzle glyph that keeps the accessible label.",
      },
    },
  },
  argTypes: {
    provider: {
      control: "text",
      description: "Provider slug, e.g. `github` or `google_maps`",
    },
    alt: {
      control: "text",
      description: "Accessible label; defaults to the provider slug",
    },
    size: {
      control: { type: "number", min: 12, max: 96, step: 4 },
      description: "Width and height in pixels",
    },
  },
  args: { provider: "github", size: 16 },
} satisfies Meta<typeof IntegrationLogo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Large: Story = { args: { provider: "notion", size: 48 } };

export const CustomLabel: Story = {
  args: { provider: "google_maps", alt: "Google Maps", size: 32 },
};

export const SlugIsScrubbed: Story = {
  args: { provider: "Google Maps", size: 32 },
  parameters: {
    docs: {
      description: {
        story:
          "Spaces, dashes and casing are normalised, so `Google Maps` resolves to `google_maps.png`.",
      },
    },
  },
};

export const Microsoft365Copilot: Story = {
  args: { provider: "microsoft_365_copilot", size: 32 },
  parameters: {
    docs: {
      description: {
        story: "The one provider mapped to a non-PNG file (`microsoft.webp`).",
      },
    },
  },
};

export const MissingLogoFallback: Story = {
  args: { provider: "acme_crm", alt: "Acme CRM", size: 32 },
  parameters: {
    docs: {
      description: {
        story:
          "No `acme_crm.png` exists, so the image fails to load and the neutral puzzle glyph takes its place, still labelled.",
      },
    },
  },
};

export const EmptySlugFallback: Story = {
  args: { provider: "???", alt: "Unknown integration", size: 32 },
  parameters: {
    docs: {
      description: {
        story:
          "A provider string that scrubs to an empty slug never requests an image and shows the glyph straight away.",
      },
    },
  },
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

export const Providers: Story = {
  render: renderProviders,
};

function renderAllSizes() {
  return (
    <div className="flex items-end gap-6">
      {SIZES.map((size) => (
        <div key={size} className="flex flex-col items-center gap-2">
          <IntegrationLogo provider="github" size={size} />
          <IntegrationLogo provider="acme_crm" alt="Acme CRM" size={size} />
          <Text variant="small" className="text-zinc-500">
            {size}px
          </Text>
        </div>
      ))}
    </div>
  );
}

function renderProviders() {
  return (
    <div className="grid grid-cols-4 gap-4">
      {[...PROVIDERS, "acme_crm"].map((provider) => (
        <div key={provider} className="flex items-center gap-2">
          <IntegrationLogo provider={provider} size={24} />
          <Text variant="small" className="text-zinc-700">
            {provider}
          </Text>
        </div>
      ))}
    </div>
  );
}
