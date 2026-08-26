import {
  getGetV2ListChatConnectionsMockHandler200,
  getGetV2ListChatConnectionsMockHandler401,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { ConnectionTier } from "@/app/api/__generated__/models/connectionTier";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { useCopilotUIStore } from "../../../../store";
import { ConnectionPicker } from "../ConnectionPicker/ConnectionPicker";

function tier(
  name: ConnectionTier["tier"],
  label: string,
  model: string | null,
): ConnectionTier {
  return { tier: name, label, selectable: true, display_model: model };
}

function offer(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
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
      tier("standard", "Balanced", "sonnet-5"),
      tier("advanced", "Advanced", "opus-5"),
    ],
    limitations: [],
    ...over,
  } as AIConnectionOffer;
}

const chatgpt = (over: Partial<AIConnectionOffer> = {}) =>
  offer({
    offer_id: "codex:cred-1",
    provider_family: "openai",
    display_name: "ChatGPT",
    auth_method: "chatgpt_oauth",
    credential_id: "cred-1",
    backed_by_label: "Your ChatGPT plan",
    is_default: false,
    tiers: [
      tier("standard", "Balanced", null),
      tier("advanced", "Advanced", null),
    ],
    limitations: ["The agent builder's chat panel always runs on AutoGPT."],
    ...over,
  });

function mockOffers(offers: AIConnectionOffer[]) {
  server.use(getGetV2ListChatConnectionsMockHandler200({ offers }));
}

beforeEach(() => {
  useCopilotUIStore.setState({
    copilotLlmAuth: null,
    copilotLlmModel: "standard",
  });
});

describe("ConnectionPicker", () => {
  it("says what backs a run on each connection", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(await screen.findByText("Your AutoGPT plan")).toBeDefined();
    expect(screen.getByText("Your ChatGPT plan")).toBeDefined();
  });

  it("falls in behind the connection the server marks default", async () => {
    mockOffers([offer({ is_default: false }), chatgpt({ is_default: true })]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Runs on ChatGPT/ }),
    ).toBeDefined();
  });

  it("names the model behind each tier where the server knows it", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByRole("radio", { name: "Balanced · sonnet-5" }),
    ).toBeDefined();
    expect(
      screen.getByRole("radio", { name: "Advanced · opus-5" }),
    ).toBeDefined();
  });

  it("gives the model its own line so a long name stays readable", async () => {
    // The model is the reason the tier row exists; sharing one line with the
    // label truncates it away on any realistic model id.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(await screen.findByText("sonnet-5")).toBeDefined();
    expect(screen.getByText("opus-5")).toBeDefined();
  });

  it("puts the chosen tier in the store, which is what the turn is sent with", async () => {
    // The picker is the only place a tier is chosen, and the store is the only
    // thing the send path reads. A regression once gated the value -- but not
    // this control -- on CHAT_MODE_OPTION, so the segment moved to Advanced
    // and the turn still ran Balanced. Landing the choice here is the contract
    // the send path depends on.
    mockOffers([offer()]);
    render(<ConnectionPicker connectionLocked />);

    await userEvent.click(
      await screen.findByRole("button", { name: /Model tier/ }),
    );
    await userEvent.click(
      await screen.findByRole("radio", { name: "Advanced · opus-5" }),
    );

    await waitFor(() =>
      expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced"),
    );
  });

  it("offers no tier choice when both tiers are the same model", async () => {
    // A single-model self-host resolves both tiers identically; choosing
    // between two identical options is a decision with no consequence.
    mockOffers([
      offer({
        tiers: [
          tier("standard", "Balanced", "ornith-1.5-9b"),
          tier("advanced", "Advanced", "ornith-1.5-9b"),
        ],
      }),
      chatgpt(),
    ]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("radiogroup", { name: "Model tier" }),
      ).toBeNull(),
    );
  });

  it("keeps the tiers when the models are merely unknown", async () => {
    // Unknown is not the same as equal — a ChatGPT connection cannot name its
    // models without a provider call, which is not a reason to hide the choice.
    mockOffers([chatgpt({ is_default: true })]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByRole("radio", { name: "Balanced" }),
    ).toBeDefined();
  });

  it("switches the connection a chat will run on", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );
    await userEvent.click(
      await screen.findByRole("radio", { name: /ChatGPT/ }),
    );

    await waitFor(() =>
      expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
        authProvider: "codex",
        credentialId: "cred-1",
      }),
    );
  });

  it("surfaces a limitation the user can actually hit", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByText(/builder's chat panel always runs on AutoGPT/),
    ).toBeDefined();
  });

  it("stays out of the way when there is nothing to choose", async () => {
    mockOffers([
      offer({
        tiers: [
          tier("standard", "Balanced", "one-model"),
          tier("advanced", "Advanced", "one-model"),
        ],
      }),
    ]);

    render(<ConnectionPicker />);

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /Runs on/ })).toBeNull(),
    );
  });

  it("still lets an underway chat change its model tier", async () => {
    // The connection is fixed when a session is created, but the tier is a
    // per-message setting -- it was changeable between turns before the two
    // controls were merged, and must stay so.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker connectionLocked />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Model tier/ }),
    );

    expect(
      await screen.findByRole("radio", { name: "Advanced · opus-5" }),
    ).toBeDefined();
  });

  it("does not offer to move an underway chat to another connection", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker connectionLocked />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Model tier/ }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("radiogroup", {
          name: "Connection this chat runs on",
        }),
      ).toBeNull(),
    );
  });

  it("stays out of an underway chat that has no tier to choose", async () => {
    mockOffers([
      offer({
        tiers: [
          tier("standard", "Balanced", "one-model"),
          tier("advanced", "Advanced", "one-model"),
        ],
      }),
      chatgpt(),
    ]);

    render(<ConnectionPicker connectionLocked />);

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /Model tier/ })).toBeNull(),
    );
  });

  it("reports a failure rather than inventing a connection", async () => {
    server.use(getGetV2ListChatConnectionsMockHandler401());

    render(<ConnectionPicker />);

    expect(
      await screen.findByLabelText("AI connections unavailable"),
    ).toBeDefined();
  });
});
