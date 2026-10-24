import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { AIConnectionOffersResponse } from "@/app/api/__generated__/models/aIConnectionOffersResponse";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { useCopilotUIStore } from "../../../../store";
import { ConnectionPicker } from "./ConnectionPicker";

// One deployment connection with two distinct tiers: there is no connection
// to choose between, so the chip folds down to the tier glyph and names the
// tier and its model in a tooltip.
const PLATFORM_OFFER: AIConnectionOffer = {
  offer_id: "platform:deployment",
  auth_provider: "platform",
  provider_family: "autogpt",
  display_name: "AutoGPT Platform",
  auth_method: "deployment",
  credential_id: null,
  backed_by_label: "Your AutoGPT plan",
  description: "New chats are backed by your AutoGPT plan.",
  state: "ready",
  selectable: true,
  is_default: true,
  tiers: [
    {
      tier: "standard",
      label: "Balanced",
      selectable: true,
      display_model: "Sonnet 5",
    },
    {
      tier: "advanced",
      label: "Advanced",
      selectable: true,
      display_model: "Opus 5",
    },
  ],
  limitations: [],
};

const offersHandler = http.get("*/api/chat/connections", () =>
  HttpResponse.json<AIConnectionOffersResponse>({ offers: [PLATFORM_OFFER] }),
);

const meta: Meta<typeof ConnectionPicker> = {
  title: "Copilot/ConnectionPicker",
  component: ConnectionPicker,
  parameters: {
    layout: "padded",
    msw: { handlers: [offersHandler] },
    docs: {
      description: {
        component:
          "Picks the connection and quality tier a turn runs on. With a single connection it shows only the tier glyph, with the tier and model in a tooltip.",
      },
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="pt-12">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const BalancedTier: Story = {
  name: "Tier chip, Balanced",
  decorators: [
    (Story) => {
      useCopilotUIStore.setState({ copilotLlmModel: "standard" });
      return <Story />;
    },
  ],
};

export const AdvancedTier: Story = {
  name: "Tier chip, Advanced",
  decorators: [
    (Story) => {
      useCopilotUIStore.setState({ copilotLlmModel: "advanced" });
      return <Story />;
    },
  ],
};
