import { getResetExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { kickoffStorageKey } from "@/app/(platform)/copilot/expertKickoff";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertResetSection } from "../ExpertResetSection";

const toastMock = vi.hoisted(() => vi.fn());
const auth = vi.hoisted(() => ({ user: { id: "user-1" } }));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

vi.mock("@/lib/auth/hooks/useAuthStore", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/auth/hooks/useAuthStore")>();
  type State = ReturnType<typeof actual.useAuthStore.getState>;
  function useAuthStore<T>(selector: (state: State) => T): T {
    return selector({
      ...actual.useAuthStore.getState(),
      user: auth.user as State["user"],
    });
  }
  return {
    ...actual,
    useAuthStore: Object.assign(useAuthStore, actual.useAuthStore),
  };
});

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: null,
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

const RESET_URL = "*/api/experts/expert-maria/reset";

beforeEach(() => {
  toastMock.mockReset();
  window.localStorage.setItem(
    kickoffStorageKey("user-1", "expert-maria"),
    "done:1:a",
  );
});

afterEach(() => {
  window.localStorage.clear();
});

function busy() {
  return http.post(RESET_URL, () =>
    HttpResponse.json(
      { detail: "Maria is still working. Reset once they finish." },
      { status: 409 },
    ),
  );
}

const DONE = {
  memory_nodes_erased: 3,
  computer_erased: true,
  soul_reset: true,
  routines_reset: 1,
  chats_archived: 2,
  approvals_cleared: 1,
};

describe("Reset expert", () => {
  test("asks first, saying what is erased, removed and kept", async () => {
    const reset = vi.fn();
    server.use(
      http.post(RESET_URL, () => {
        reset();
        return HttpResponse.json({});
      }),
    );
    const user = userEvent.setup();
    render(<ExpertResetSection expert={maria} />);

    await user.click(screen.getByTestId("expert-reset-button"));

    const dialog = await screen.findByRole("dialog", { name: "Reset Maria?" });
    const text = dialog.textContent;
    expect(text).toContain("the day you hired them");
    expect(text).toContain("can't be undone");
    expect(text).toContain("Everything Maria remembers");
    expect(text).toContain("Maria's computer and the notes on it");
    expect(text).toContain("Routines you switched on are switched off");
    expect(text).toContain("Removed");
    expect(text).toContain("can't be opened again");
    expect(text).toContain("Approvals from those chats waiting in Needs you");
    expect(text).toContain("The skills Maria has learned");
    expect(text).toContain("Their workflows, and the schedules on them");
    expect(text).toContain("what the rest of your team knows");
    expect(text).not.toMatch(/archived/i);
    expect(reset).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "Reset Maria?" })).toBeNull(),
    );
    expect(reset).not.toHaveBeenCalled();
  });

  test("tells a created expert's owner what a reset of it keeps", async () => {
    const user = userEvent.setup();
    render(
      <ExpertResetSection expert={{ ...maria, source_template_id: null }} />,
    );

    expect(screen.getByText(/the day you created them/)).toBeDefined();
    await user.click(screen.getByTestId("expert-reset-button"));

    const dialog = await screen.findByRole("dialog", { name: "Reset Maria?" });
    const text = dialog.textContent;
    expect(text).toContain("the day you created them");
    expect(text).not.toContain("the day you hired them");
    expect(text).not.toContain("Your edits to their soul and routines");
    expect(text).not.toContain("Routines you switched on are switched off");
    expect(text).toContain(
      "Their soul and routines, each routine still on or off as it is now",
    );
  });

  test("shows that Maria is reset, with a way to chat with her", async () => {
    server.use(getResetExpertMockHandler(DONE));
    const user = userEvent.setup();
    render(<ExpertResetSection expert={maria} />);

    await user.click(screen.getByTestId("expert-reset-button"));
    await user.click(await screen.findByTestId("reset-expert-confirm"));

    const done = await screen.findByRole("dialog", { name: "Maria is reset" });
    expect(done.textContent).toContain("your old chats with them are removed");
    expect(done.textContent).toContain(
      "Chat with Maria and they'll introduce themselves again.",
    );
    // A new chat that opens with Maria introducing herself, as after a hire.
    expect(
      screen
        .getByRole("link", { name: "Chat with Maria" })
        .getAttribute("href"),
    ).toBe("/copilot?expertId=expert-maria&kickoff=1");
    expect(
      window.localStorage.getItem(kickoffStorageKey("user-1", "expert-maria")),
    ).toBeNull();
    expect(toastMock).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Maria is reset" }),
      ).toBeNull(),
    );

    // Opening it again starts over at the question.
    await user.click(screen.getByTestId("expert-reset-button"));
    expect(
      await screen.findByRole("dialog", { name: "Reset Maria?" }),
    ).toBeDefined();
  });

  test("says Maria is still working, and resets on a retry once she is not", async () => {
    server.use(busy());
    const user = userEvent.setup();
    render(<ExpertResetSection expert={maria} />);

    await user.click(screen.getByTestId("expert-reset-button"));
    await user.click(await screen.findByTestId("reset-expert-confirm"));

    const waiting = await screen.findByRole("dialog", {
      name: "Maria is still working",
    });
    expect(waiting.textContent).toContain("nothing was reset");
    expect(toastMock).not.toHaveBeenCalled();
    expect(
      window.localStorage.getItem(kickoffStorageKey("user-1", "expert-maria")),
    ).toBe("done:1:a");

    server.use(getResetExpertMockHandler(DONE));
    await user.click(screen.getByTestId("reset-expert-retry"));

    expect(
      await screen.findByRole("dialog", { name: "Maria is reset" }),
    ).toBeDefined();
  });

  test("closes from the busy state without resetting", async () => {
    const reset = vi.fn();
    server.use(
      http.post(RESET_URL, () => {
        reset();
        return HttpResponse.json(
          { detail: "Maria is still working. Reset once they finish." },
          { status: 409 },
        );
      }),
    );
    const user = userEvent.setup();
    render(<ExpertResetSection expert={maria} />);

    await user.click(screen.getByTestId("expert-reset-button"));
    await user.click(await screen.findByTestId("reset-expert-confirm"));
    await screen.findByRole("dialog", { name: "Maria is still working" });

    await user.click(screen.getByTestId("reset-expert-busy-close"));

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Maria is still working" }),
      ).toBeNull(),
    );
    expect(reset).toHaveBeenCalledTimes(1);
  });

  test("says so when the reset did not finish", async () => {
    server.use(
      http.post(RESET_URL, () =>
        HttpResponse.json(
          { detail: "The expert couldn't be fully reset. Try again." },
          { status: 503 },
        ),
      ),
    );
    const user = userEvent.setup();
    render(<ExpertResetSection expert={maria} />);

    await user.click(screen.getByTestId("expert-reset-button"));
    await user.click(await screen.findByTestId("reset-expert-confirm"));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Could not finish resetting Maria",
          variant: "destructive",
        }),
      ),
    );
    expect(screen.getByRole("dialog", { name: "Reset Maria?" })).toBeTruthy();
  });
});
