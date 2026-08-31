import {
  getGetV2ListChatConnectionsMockHandler200,
  getGetV2ListChatConnectionsMockHandler401,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { ConnectionTier } from "@/app/api/__generated__/models/connectionTier";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
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

const locked = (over: Partial<AIConnectionOffer> = {}) =>
  offer({
    offer_id: "codex:locked",
    provider_family: "openai",
    display_name: "ChatGPT",
    auth_method: "chatgpt_oauth",
    credential_id: null,
    backed_by_label: "Your ChatGPT plan",
    state: "locked",
    selectable: false,
    is_default: false,
    tiers: [],
    limitations: [],
    description:
      "Run chats on a ChatGPT plan you already pay for, spending no AutoGPT credits.",
    lock_reason: "A Max plan or higher is required to use ChatGPT.",
    unlock_href: "/settings/billing",
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

  it("puts the chosen tier in the store", async () => {
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

  it("shows the server default without claiming it as a choice", async () => {
    // The store is what the create call treats as an override, and it has no
    // way back to null. Writing the default into it on render made every
    // later chat inherit whatever this one displayed, and made a default
    // changed in Settings unable to take over. Displaying is not choosing.
    mockOffers([offer(), chatgpt()]);
    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /AutoGPT Platform/ }),
    ).toBeDefined();
    expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();
  });

  it("records the connection once the user actually picks one", async () => {
    mockOffers([offer(), chatgpt()]);
    render(<ConnectionPicker />);

    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );
    await userEvent.click(
      await screen.findByRole("radio", { name: /ChatGPT/ }),
    );

    await waitFor(() =>
      expect(useCopilotUIStore.getState().copilotLlmAuth).toMatchObject({
        authProvider: "codex",
      }),
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

  it("says why a connection the plan excludes is unavailable", async () => {
    mockOffers([offer(), locked()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(await screen.findByText("ChatGPT")).toBeDefined();
    expect(
      screen.getByText("A Max plan or higher is required to use ChatGPT."),
    ).toBeDefined();
    expect(screen.getByRole("link", { name: "See plans" })).toBeDefined();
  });

  it("keeps a locked-only offer visible", async () => {
    mockOffers([locked()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on Choose connection/ }),
    );

    expect(
      await screen.findByText(
        "A Max plan or higher is required to use ChatGPT.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("link", { name: "See plans" })).toBeDefined();
  });

  it("keeps a locked-only offer visible after an underway chat loses access", async () => {
    mockOffers([locked()]);

    render(<ConnectionPicker connectionLocked />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on Choose connection/ }),
    );

    expect(
      await screen.findByText(
        "A Max plan or higher is required to use ChatGPT.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("link", { name: "See plans" })).toBeDefined();
  });

  it("spends a locked row on the benefit, not on a plan the user may lack", async () => {
    mockOffers([offer(), locked()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByText(/spending no AutoGPT credits/),
    ).toBeDefined();
    // "Your ChatGPT plan" above "a Max plan is required" reads as two
    // different plans, and presumes one they may not have.
    expect(screen.queryByText("Your ChatGPT plan")).toBeNull();
  });

  it("keeps named models visible on a locked connection", async () => {
    mockOffers([
      offer(),
      locked({
        tiers: [
          tier("standard", "Balanced", "gpt-5.6-terra"),
          tier("advanced", "Advanced", "gpt-5.6-sol"),
        ],
      }),
    ]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByText(
        "Balanced: gpt-5.6-terra · Advanced: gpt-5.6-sol",
      ),
    ).toBeDefined();
  });

  it("does not offer a locked connection as something to pick", async () => {
    // Rendering it as a radio would invite a click that cannot take effect.
    mockOffers([offer(), locked()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );
    await screen.findByText("A Max plan or higher is required to use ChatGPT.");

    expect(screen.queryByRole("radio", { name: /ChatGPT/ })).toBeNull();
  });

  it("never falls back onto a locked connection", async () => {
    // is_default should never point at a locked offer, but the client is not
    // the place to find out the hard way.
    mockOffers([offer({ is_default: false }), locked({ is_default: true })]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Runs on AutoGPT Platform/ }),
    ).toBeDefined();
    // What it lands on is what it shows: a locked offer is never the one the
    // chip names. Nothing is written into the store either, because showing a
    // default is not the user choosing it.
    expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();
  });

  it("names both models on a connection before you switch to it", async () => {
    // Otherwise you have to select a connection to discover what it runs,
    // which is the wrong order.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByText("Balanced: sonnet-5 · Advanced: opus-5"),
    ).toBeDefined();
  });

  it("badges a connection the user linked themselves", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(await screen.findByText("Connected")).toBeDefined();
    // The deployment route is not something anyone connected.
    expect(screen.getAllByText("Connected")).toHaveLength(1);
  });

  it("shows a tier the plan excludes as locked rather than hiding it", async () => {
    // The row is the upgrade reason; hiding it removes what it exists for.
    mockOffers([
      offer({
        tiers: [
          tier("standard", "Balanced", "sonnet-5"),
          {
            ...tier("advanced", "Advanced", "opus-5"),
            selectable: false,
            lock_reason: "A Max plan or higher is required for Advanced.",
          },
        ],
      }),
      chatgpt(),
    ]);

    render(<ConnectionPicker />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Runs on/ }),
    );

    expect(
      await screen.findByText("A Max plan or higher is required for Advanced."),
    ).toBeDefined();
    // Still named, so the user sees what they would get.
    expect(screen.getByText("opus-5")).toBeDefined();
    // Scoped to the tier group: the connection row's own name now contains
    // "Advanced: opus-5" from its tier summary, so a page-wide query matches it.
    const tierGroup = screen.getByRole("radiogroup", { name: "Model tier" });
    expect(
      within(tierGroup).queryByRole("radio", { name: /Advanced/ }),
    ).toBeNull();
    expect(within(tierGroup).getAllByRole("radio")).toHaveLength(1);
  });

  it("reports a failure rather than inventing a connection", async () => {
    server.use(getGetV2ListChatConnectionsMockHandler401());

    render(<ConnectionPicker />);

    expect(
      await screen.findByLabelText("AI connections unavailable"),
    ).toBeDefined();
  });
});
