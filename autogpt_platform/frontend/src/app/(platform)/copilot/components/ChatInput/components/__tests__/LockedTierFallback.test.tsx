import { getGetV2ListChatConnectionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";

import { useCopilotUIStore } from "../../../../store";
import { ConnectionPicker } from "../ConnectionPicker/ConnectionPicker";

const ADVANCED_LOCK = "A Max plan or higher is required for Advanced.";

function platformOffer(advancedLocked: boolean): AIConnectionOffer {
  return {
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
        display_model: "sonnet-5",
      },
      {
        tier: "advanced",
        label: "Advanced",
        selectable: !advancedLocked,
        display_model: "opus-5",
        lock_reason: advancedLocked ? ADVANCED_LOCK : null,
      },
    ],
    limitations: [],
  } as AIConnectionOffer;
}

beforeEach(() => {
  window.localStorage.clear();
});

describe("a stored tier the server has locked", () => {
  it("falls back to Balanced, in the store and in storage, so sends stop being refused", async () => {
    useCopilotUIStore.getState().setCopilotLlmModel("advanced");
    server.use(
      getGetV2ListChatConnectionsMockHandler200({
        offers: [platformOffer(true)],
      }),
    );

    render(<ConnectionPicker />);

    await waitFor(() =>
      expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard"),
    );
    expect(window.localStorage.getItem("copilot-model")).toBe("standard");
    expect(
      await screen.findByRole("button", { name: /Balanced/ }),
    ).toBeDefined();
  });

  it("keeps Advanced when the server offers it", async () => {
    useCopilotUIStore.getState().setCopilotLlmModel("advanced");
    server.use(
      getGetV2ListChatConnectionsMockHandler200({
        offers: [platformOffer(false)],
      }),
    );

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Advanced/ }),
    ).toBeDefined();
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced");
  });

  it("leaves a stored Balanced alone while Advanced is locked", async () => {
    useCopilotUIStore.getState().setCopilotLlmModel("standard");
    server.use(
      getGetV2ListChatConnectionsMockHandler200({
        offers: [platformOffer(true)],
      }),
    );

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Balanced/ }),
    ).toBeDefined();
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
    expect(window.localStorage.getItem("copilot-model")).toBe("standard");
  });
});
