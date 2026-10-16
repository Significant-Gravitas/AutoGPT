import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
} from "./sse-helpers";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => ({
  ...(await importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));

const UNAVAILABLE = /this chat isn't available on this account/i;

function sessionError(status: number) {
  let calls = 0;
  const handler = http.get(`*/api/chat/sessions/${TEST_SESSION_ID}`, () => {
    calls += 1;
    return HttpResponse.json(
      { detail: `Session ${TEST_SESSION_ID} not found.` },
      { status },
    );
  });
  return { handler, calls: () => calls };
}

beforeEach(() => {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
});

afterEach(() => {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
});

describe("a chat link the user cannot open", () => {
  it("says the chat is unavailable and offers a new chat on a 404", async () => {
    const { handler } = sessionError(404);
    renderHost({ sessionResponse: handler });

    expect(await screen.findByText(UNAVAILABLE)).toBeDefined();
    expect(
      screen.queryByPlaceholderText("What else can I help with?"),
    ).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /new chat/i }));
    await waitFor(() => expect(screen.queryByText(UNAVAILABLE)).toBeNull());
  });

  it("keeps a server error on the existing path rather than calling the chat unavailable", async () => {
    const { handler, calls } = sessionError(500);
    renderHost({ sessionResponse: handler });

    await waitFor(() => expect(calls()).toBeGreaterThan(0));
    expect(
      await screen.findByPlaceholderText("What else can I help with?"),
    ).toBeDefined();
    expect(screen.queryByText(UNAVAILABLE)).toBeNull();
  });
});
