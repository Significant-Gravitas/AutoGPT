import { getGetV2GetSuggestedPromptsMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { PERSONAS } from "../personas";

afterEach(cleanup);

beforeEach(() => {
  window.history.replaceState(null, "", "/copilot");
  server.use(
    getGetV2GetSuggestedPromptsMockHandler({
      themes: [],
    }),
  );
});

function renderEmptySession() {
  render(
    <EmptySession
      inputLayoutId="test-input"
      isCreatingSession={false}
      onCreateSession={vi.fn()}
      onSend={vi.fn()}
    />,
  );
}

function openDial() {
  fireEvent.click(screen.getByLabelText(/Change persona/));
}

describe("EmptySession persona selection", () => {
  it("greets as Autopilot by default", () => {
    renderEmptySession();
    expect(screen.getByText("Autopilot")).toBeTruthy();
    expect(screen.getByText(/your generalist/)).toBeTruthy();
    expect(screen.getByPlaceholderText("Automate Anything...")).toBeTruthy();
  });

  it("opens the persona dial from the avatar", () => {
    renderEmptySession();
    openDial();
    expect(screen.getByLabelText("Search personas")).toBeTruthy();
    expect(screen.getAllByRole("option").length).toBe(PERSONAS.length);
  });

  it("re-greets as the picked persona and mirrors it into the URL", async () => {
    renderEmptySession();
    openDial();
    const input = screen.getByLabelText("Search personas");
    fireEvent.change(input, { target: { value: "blaze" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(screen.getByText("Blaze")).toBeTruthy();
    });
    expect(screen.getByText(/your marketer/)).toBeTruthy();
    expect(window.location.search).toBe("?persona=blaze");
    expect(screen.queryByLabelText("Search personas")).toBeNull();
  });

  it("restores the persona from the URL on load", async () => {
    window.history.replaceState(null, "", "/copilot?persona=wren");
    renderEmptySession();
    await waitFor(() => {
      expect(screen.getByText("Wren")).toBeTruthy();
    });
  });

  it("closes the dial with Escape without changing the persona", () => {
    renderEmptySession();
    openDial();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByLabelText("Search personas")).toBeNull();
    expect(screen.getByText("Autopilot")).toBeTruthy();
  });
});
