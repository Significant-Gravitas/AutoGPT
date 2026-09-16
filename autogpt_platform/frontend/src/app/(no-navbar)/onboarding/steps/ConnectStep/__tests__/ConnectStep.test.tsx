import {
  getGetV2ListChatConnectionsMockHandler200,
  getGetV2ListProviderModelTiersMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { ProviderTiers } from "@/app/api/__generated__/models/providerTiers";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useOnboardingWizardStore } from "../../../store";
import { ConnectStep } from "../ConnectStep";
import { hasLinkedSubscription, linkedModelsSentence } from "../helpers";

const connect = vi.fn();
vi.mock(
  "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect",
  () => ({
    useOAuthConnect: () => ({ connect, isPending: false }),
  }),
);

function offer(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return {
    offer_id: "platform:deployment",
    provider_family: "autogpt",
    display_name: "Self-hosted chat",
    auth_method: "deployment",
    credential_id: null,
    backed_by_label: "This server's chat provider",
    description: "New chats are backed by the chat provider on this server.",
    state: "ready",
    selectable: true,
    is_default: true,
    tiers: [],
    limitations: [],
    lock_reason: null,
    unlock_href: null,
    ...over,
  } as AIConnectionOffer;
}

function chatgpt(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return offer({
    offer_id: "codex:cred-1",
    provider_family: "openai",
    display_name: "ChatGPT",
    auth_method: "chatgpt_oauth",
    credential_id: "cred-1",
    is_default: false,
    tiers: [
      {
        tier: "standard",
        label: "Balanced",
        selectable: true,
        display_model: "GPT-5.6 Terra",
      },
      {
        tier: "advanced",
        label: "Advanced",
        selectable: true,
        display_model: "GPT-5.6 Sol",
      },
    ],
    ...over,
  });
}

function chatgptTiers(): ProviderTiers {
  return {
    provider_family: "openai",
    display_name: "ChatGPT",
    tiers: [
      { tier: "standard", label: "Balanced", display_model: "GPT-5.6 Terra" },
      { tier: "advanced", label: "Advanced", display_model: "GPT-5.6 Sol" },
    ],
  } as ProviderTiers;
}

function mockOffers(
  offers: AIConnectionOffer[],
  providers: ProviderTiers[] = [],
) {
  server.use(
    getGetV2ListChatConnectionsMockHandler200({ offers }),
    getGetV2ListProviderModelTiersMockHandler200({ providers }),
  );
}

describe("ConnectStep", () => {
  beforeEach(() => {
    connect.mockClear();
    useOnboardingWizardStore.setState({ currentStep: 1 });
  });

  it("leads with the zero-config path", async () => {
    mockOffers([offer()]);

    render(<ConnectStep />);

    expect(
      await screen.findByRole("button", { name: /Sign in with ChatGPT/ }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: /Skip for now/ })).toBeDefined();
  });

  it("can be skipped, because API keys are a legitimate answer", async () => {
    // A wizard that cannot be passed without linking an account would make
    // the advanced path a dead end rather than an alternative. The control
    // is named for what it does -- it skips the step; it cannot add a key --
    // and the line beside it says where keys actually live.
    mockOffers([offer()]);

    render(<ConnectStep />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Skip for now/ }),
    );

    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
    expect(connect).not.toHaveBeenCalled();
  });

  it("stops asking once a subscription is linked", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectStep />);

    expect(
      await screen.findByText(/Your ChatGPT plan is connected/),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Sign in with ChatGPT/ })).toBe(
      null,
    );
  });

  it("says a Claude subscription cannot be linked, rather than leaving it to be discovered", async () => {
    mockOffers([offer()]);

    render(<ConnectStep />);

    expect(
      await screen.findByText(/Claude subscriptions can't be linked/),
    ).toBeDefined();
  });

  it("names what you get, from the catalog", async () => {
    mockOffers([offer()], [chatgptTiers()]);

    render(<ConnectStep />);

    expect(
      await screen.findByText(
        /GPT-5\.6 Terra \(Balanced\) and GPT-5\.6 Sol \(Advanced\)/,
      ),
    ).toBeDefined();
  });
});

describe("ConnectStep helpers", () => {
  it("does not count the deployment's own provider as a linked subscription", () => {
    // It is a connection, but it is the one that exists because someone put
    // an API key in a file -- which is what this step offers a way around.
    expect(hasLinkedSubscription([offer()])).toBe(false);
    expect(hasLinkedSubscription([offer(), chatgpt()])).toBe(true);
  });

  it("names the models from the catalog rather than hardcoding them", () => {
    expect(linkedModelsSentence([chatgptTiers()])).toBe(
      "GPT-5.6 Terra (Balanced) and GPT-5.6 Sol (Advanced)",
    );
  });

  it("reads them from provider tiers, not from the user's connections", () => {
    // The whole point of this screen is that the user has not connected yet,
    // so ChatGPT is absent from their offers and this sentence would always
    // have come out empty if it read them.
    expect(linkedModelsSentence([])).toBe("");
  });

  it("says nothing when the server named no models", () => {
    expect(
      linkedModelsSentence([{ ...chatgptTiers(), tiers: [] } as ProviderTiers]),
    ).toBe("");
    expect(linkedModelsSentence(undefined)).toBe("");
  });
});
