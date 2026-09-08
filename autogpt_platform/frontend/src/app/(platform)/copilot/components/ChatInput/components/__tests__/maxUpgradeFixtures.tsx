import { getGetV2ListChatConnectionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { server } from "@/mocks/mock-server";
import { screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
export async function openPicker() {
  await userEvent.click(
    await screen.findByRole("button", { name: /— change/ }),
  );
}

export function deploymentOffer(
  overrides: Partial<AIConnectionOffer> = {},
): AIConnectionOffer {
  return {
    offer_id: "platform:deployment",
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
        display_model: "sonnet-server",
      },
      {
        tier: "advanced",
        label: "Advanced",
        selectable: false,
        display_model: "opus-server",
        lock_reason: "A Max plan or higher is required for Advanced.",
      },
    ],
    limitations: [],
    ...overrides,
  };
}

export function mockMaxUpgrade(offers = [deploymentOffer()]) {
  server.use(getGetV2ListChatConnectionsMockHandler200({ offers }));
}
