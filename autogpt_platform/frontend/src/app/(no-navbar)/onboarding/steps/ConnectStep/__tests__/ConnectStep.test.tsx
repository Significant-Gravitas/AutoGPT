import { getGetV2ListChatConnectionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetV2ListChatConnectionsMockHandler200,
  getGetV2ListProviderModelTiersMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { ProviderTiers } from "@/app/api/__generated__/models/providerTiers";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { type QueryClient, useQueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useOnboardingWizardStore } from "../../../store";
import { ConnectStep } from "../ConnectStep";
import { hasLinkedSubscription, linkedModelsSentence } from "../helpers";

const connect = vi.fn();
let onConnected: (() => void) | undefined;
vi.mock(
  "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect",
  () => ({
    useOAuthConnect: ({ onSuccess }: { onSuccess?: () => void }) => {
      onConnected = onSuccess;
      return { connect, isPending: false };
    },
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

// Exposes the test QueryClient so a test can wait for a query to settle.
let queryClient: QueryClient | null = null;
function QueryClientProbe() {
  queryClient = useQueryClient();
  return null;
}

describe("ConnectStep", () => {
  beforeEach(() => {
    connect.mockClear();
    useOnboardingWizardStore.setState({ currentStep: 3 });
  });

  it("offers ChatGPT as a box and names the rest as coming soon", async () => {
    mockOffers([offer()]);

    render(<ConnectStep />);

    expect(
      await screen.findByRole("button", { name: /ChatGPT/ }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Next" })).toBeDefined();
    // Upcoming providers are shown, not clickable: each needs its own
    // adapter and approval before it can become a real box.
    expect(screen.getByText("Grok")).toBeDefined();
    expect(screen.getByText("GitHub Copilot")).toBeDefined();
    expect(screen.getAllByText("Coming soon")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: /Grok/ })).toBeNull();
  });

  it("starts the ChatGPT sign-in from its box", async () => {
    mockOffers([offer()]);

    render(<ConnectStep />);
    await userEvent.click(
      await screen.findByRole("button", { name: /ChatGPT/ }),
    );

    expect(connect).toHaveBeenCalledTimes(1);
  });

  it("can be passed with Next, because API keys are a legitimate answer", async () => {
    // A wizard that cannot be passed without linking an account would make
    // the advanced path a dead end rather than an alternative. The control
    // is named for what it does -- it skips the step; it cannot add a key --
    // and the line beside it says where keys actually live.
    mockOffers([offer()]);

    render(<ConnectStep />);
    await userEvent.click(await screen.findByRole("button", { name: "Next" }));

    expect(useOnboardingWizardStore.getState().currentStep).toBe(4);
    expect(connect).not.toHaveBeenCalled();
  });

  it("stays on the step after a successful sign-in and shows it linked", async () => {
    mockOffers([offer()]);

    render(
      <>
        <QueryClientProbe />
        <ConnectStep />
      </>,
    );
    await screen.findByRole("button", { name: /ChatGPT/ });
    // The click can only come after the page has settled: a refetch asked
    // for while the first request is still in flight is folded into it.
    await waitFor(() =>
      expect(
        queryClient?.getQueryState(getGetV2ListChatConnectionsQueryKey())
          ?.fetchStatus,
      ).toBe("idle"),
    );

    // The sign-in lands; the server now lists the ChatGPT connection.
    mockOffers([offer(), chatgpt()]);
    onConnected?.();

    expect(await screen.findByText("Connected")).toBeDefined();
    expect(useOnboardingWizardStore.getState().currentStep).toBe(3);
  });

  it("stops asking once a subscription is linked", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectStep />);

    expect(await screen.findByText("Connected")).toBeDefined();
    expect(screen.queryByRole("button", { name: /ChatGPT/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Next" })).toBeDefined();
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
