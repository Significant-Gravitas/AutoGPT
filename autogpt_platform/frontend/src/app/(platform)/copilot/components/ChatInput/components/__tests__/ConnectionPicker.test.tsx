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

/**
 * The picker's trigger, whichever thing it is currently labelled with.
 *
 * The chip names the connection when there is one to choose, and the tier when
 * there is not, so matching either label alone would tie every test that merely
 * needs to open the popover to the label rule. Both end in "— change".
 */
function openPicker() {
  return screen.findByRole("button", { name: /— change/ });
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
    await userEvent.click(await openPicker());

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
    await userEvent.click(await openPicker());

    expect(
      await screen.findByRole("radio", { name: "Balanced · sonnet-5" }),
    ).toBeDefined();
    expect(
      screen.getByRole("radio", { name: "Advanced · opus-5" }),
    ).toBeDefined();
  });

  it("names the model inside each tier segment", async () => {
    // The model is the reason the tier control exists, so it is on the control
    // rather than a line under it. This fits only because the model arrives as
    // a display name; while it arrived as "anthropic/claude-sonnet-5" the label
    // had to wrap, which is why this once asserted a separate line.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());

    const tiers = await screen.findByRole("radiogroup", { name: "Model tier" });
    expect(
      within(tiers).getByRole("radio", { name: "Balanced · sonnet-5" }),
    ).toBeDefined();
    expect(
      within(tiers).getByRole("radio", { name: "Advanced · opus-5" }),
    ).toBeDefined();
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

    await userEvent.click(await openPicker());
    await userEvent.click(
      await screen.findByRole("radio", { name: /ChatGPT/ }),
    );

    await waitFor(() =>
      expect(useCopilotUIStore.getState().copilotLlmAuth).toMatchObject({
        authProvider: "codex",
      }),
    );
  });

  it("moves and selects between connections with the arrow keys", async () => {
    mockOffers([offer(), chatgpt()]);
    render(<ConnectionPicker />);

    await userEvent.click(await openPicker());
    const connections = await screen.findByRole("radiogroup", {
      name: "Connection this chat runs on",
    });
    const platform = within(connections).getByRole("radio", {
      name: /AutoGPT Platform/,
    });
    const linked = within(connections).getByRole("radio", {
      name: /ChatGPT/,
    });

    platform.focus();
    await userEvent.keyboard("x");
    expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();

    await userEvent.keyboard("{ArrowDown}");

    await waitFor(() => {
      expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
        authProvider: "codex",
        credentialId: "cred-1",
      });
      expect(document.activeElement).toBe(linked);
    });
  });

  it("is one tab stop, and the arrow keys move and select within it", async () => {
    // A radio group is not a list of buttons that say role="radio". Tab moves
    // past the whole group; the arrows move within it and select as they go.
    // Previously every option was its own tab stop and the arrows did nothing,
    // so a keyboard user could not operate the control the way its role
    // promised.
    mockOffers([offer()]);
    render(<ConnectionPicker connectionLocked />);

    await userEvent.click(
      await screen.findByRole("button", { name: /Model tier/ }),
    );
    const tiers = await screen.findByRole("radiogroup", { name: "Model tier" });
    const balanced = within(tiers).getByRole("radio", {
      name: "Balanced · sonnet-5",
    });
    const advanced = within(tiers).getByRole("radio", {
      name: "Advanced · opus-5",
    });

    expect(balanced.getAttribute("tabindex")).toBe("0");
    expect(advanced.getAttribute("tabindex")).toBe("-1");

    balanced.focus();
    await userEvent.keyboard("{ArrowRight}");

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
    await userEvent.click(await openPicker());

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
    await userEvent.click(await openPicker());

    expect(
      await screen.findByRole("radio", { name: "Balanced" }),
    ).toBeDefined();
  });

  it("switches the connection a chat will run on", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());
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
    // It waits behind the row's info mark rather than spelling itself out
    // beside the choice, but it is still reachable without leaving the popover.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());

    const rows = await screen.findByRole("radiogroup", {
      name: "Connection this chat runs on",
    });
    await userEvent.hover(within(rows).getByLabelText("More information"));

    expect(
      (
        await screen.findAllByText(
          /builder's chat panel always runs on AutoGPT/,
        )
      ).length,
    ).toBeGreaterThan(0);
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
    await userEvent.click(await openPicker());

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
    await userEvent.click(await openPicker());

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
    await userEvent.click(await openPicker());

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
    await userEvent.click(await openPicker());
    await screen.findByText("A Max plan or higher is required to use ChatGPT.");

    expect(screen.queryByRole("radio", { name: /ChatGPT/ })).toBeNull();
  });

  it("never falls back onto a locked connection", async () => {
    // is_default should never point at a locked offer, but the client is not
    // the place to find out the hard way.
    mockOffers([offer({ is_default: false }), locked({ is_default: true })]);

    render(<ConnectionPicker />);

    await userEvent.click(await openPicker());
    // What it lands on is what it marks selected: never the locked one.
    const landed = await screen.findByRole("radio", {
      name: /AutoGPT Platform/,
    });
    expect(landed.getAttribute("aria-checked")).toBe("true");
    // What it lands on is what it shows: a locked offer is never the one the
    // chip names. Nothing is written into the store either, because showing a
    // default is not the user choosing it.
    expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();
  });

  it("leaves the models to the tier toggle on a selectable connection", async () => {
    // The toggle names them a few pixels below, so repeating them per row only
    // crowds the choice the rows exist to present.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());

    await screen.findByRole("radio", { name: /AutoGPT Platform/ });
    expect(
      screen.queryByText("Balanced: sonnet-5 · Advanced: opus-5"),
    ).toBeNull();
    expect(
      screen.getByRole("radio", { name: "Balanced \u00b7 sonnet-5" }),
    ).toBeDefined();
  });

  it("badges a connection the user linked themselves", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());

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
    await userEvent.click(await openPicker());

    expect(
      await screen.findByText(/A Max plan or higher is required for Advanced/),
    ).toBeDefined();
    // Still named, so the user sees what they would get. Scoped to the tier
    // control: the connection row's own summary also mentions Advanced.
    const tierGroup = screen.getByRole("radiogroup", { name: "Model tier" });
    expect(within(tierGroup).getByText(/opus-5/)).toBeDefined();
    // It remains a member of the radiogroup for assistive technology, but is
    // exposed as disabled rather than offered as a choice that can happen.
    const lockedTier = within(tierGroup).getByRole("radio", {
      name: /Advanced/,
    });
    expect(lockedTier.getAttribute("aria-disabled")).toBe("true");
    expect(lockedTier.getAttribute("aria-checked")).toBe("false");
    expect(within(tierGroup).getAllByRole("radio")).toHaveLength(2);
  });

  it("offers to link ChatGPT when the user has no ChatGPT at all", async () => {
    // Otherwise the one control about connections cannot make one, and the
    // user has to find Settings to act on what they are already looking at.
    mockOffers([offer()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());

    expect(
      await screen.findByRole("button", {
        name: /Connect a ChatGPT subscription/,
      }),
    ).toBeDefined();
  });

  it("lets a keyboard user read a connection's limitations", async () => {
    // The mark is a real button rather than the icon itself: bound to an SVG
    // the tooltip opened on hover only, so the notes it holds were reachable
    // with a mouse and by no other means.
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());

    const rows = await screen.findByRole("radiogroup", {
      name: "Connection this chat runs on",
    });
    within(rows).getByRole("button", { name: "More information" }).focus();

    expect(
      (
        await screen.findAllByText(
          /builder's chat panel always runs on AutoGPT/,
        )
      ).length,
    ).toBeGreaterThan(0);
  });

  it("sends a user with no connections at all to Settings", async () => {
    // A successful response with nothing in it leaves no popover to hang a
    // connect row on, and no routes at all is a bigger problem than an
    // unlinked ChatGPT, so the way out is the whole control.
    mockOffers([]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByLabelText("Set up an AI connection"),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /Connect a ChatGPT subscription/ }),
    ).toBeNull();
  });

  it("does not offer to link a ChatGPT the plan does not include", async () => {
    // The locked row already says what the next step is, and it is buying a
    // plan rather than signing in — an invitation to connect would send the
    // user into a flow the server can only refuse.
    mockOffers([offer(), locked()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());
    await screen.findByText("A Max plan or higher is required to use ChatGPT.");

    expect(
      screen.queryByRole("button", { name: /Connect a ChatGPT subscription/ }),
    ).toBeNull();
  });

  it("does not offer to link a ChatGPT that is already linked", async () => {
    mockOffers([offer(), chatgpt()]);

    render(<ConnectionPicker />);
    await userEvent.click(await openPicker());
    await screen.findByRole("radio", { name: /ChatGPT/ });

    expect(
      screen.queryByRole("button", { name: /Connect a ChatGPT subscription/ }),
    ).toBeNull();
  });

  it("reports a failure rather than inventing a connection", async () => {
    server.use(getGetV2ListChatConnectionsMockHandler401());

    render(<ConnectionPicker />);

    expect(
      await screen.findByLabelText("AI connections unavailable"),
    ).toBeDefined();
  });

  it("says whose plan pays when the answer is the user's own", async () => {
    // The point of labelling by payer is that spending your own subscription
    // is worth noticing before you send, not after.
    mockOffers([offer({ is_default: false }), chatgpt({ is_default: true })]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", {
        name: /Runs on ChatGPT . your plan/,
      }),
    ).toBeDefined();
  });

  it("does not claim a platform connection is the user's own plan", async () => {
    mockOffers([offer({ is_default: true }), chatgpt({ is_default: false })]);

    render(<ConnectionPicker />);

    const trigger = await screen.findByRole("button", { name: /Runs on/ });
    expect(trigger.getAttribute("aria-label")).not.toMatch(/your plan/);
  });

  it("shows the tier when there is no connection to choose", async () => {
    // Naming the only connection teaches a hosted user nothing — they would
    // read "AutoGPT Platform" on every chat. The tier is the live choice.
    mockOffers([chatgpt({ is_default: true })]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Model tier Balanced/ }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Runs on/ })).toBeNull();
  });

  it("names the connection once there is more than one", async () => {
    mockOffers([offer({ is_default: true }), chatgpt()]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Runs on AutoGPT Platform/ }),
    ).toBeDefined();
  });

  it("treats a locked alternative as no choice at all", async () => {
    // A row the user cannot pick is an explanation, not an option, so the chip
    // stays on the tier — but the row still earns its place in the popover.
    mockOffers([offer({ is_default: true }), locked()]);

    render(<ConnectionPicker />);

    expect(
      await screen.findByRole("button", { name: /Model tier/ }),
    ).toBeDefined();
    await userEvent.click(await openPicker());
    expect(screen.getByText("ChatGPT")).toBeDefined();
  });

  it("marks the chip with the tier it will run, whatever it is labelled", async () => {
    // The chip names either the connection or the tier depending on what is
    // still open, but the tier applies to the next turn either way, so its
    // glyph is on the chip in both. A key was once here instead, which against
    // "Balanced" labelled reasoning depth as an account.
    mockOffers([chatgpt({ is_default: true })]);

    render(<ConnectionPicker />);
    const trigger = await screen.findByRole("button", { name: /Model tier/ });

    // Naming only the tier, the chip folds down to the glyph alone: no label
    // and no chevron, so the one SVG is the tier.
    expect(trigger.querySelectorAll("svg")).toHaveLength(1);
  });
});
