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

vi.mock("@/lib/auth/hooks/useAuthStore", () => ({
  useAuthStore: (selector: (state: typeof auth) => unknown) => selector(auth),
}));

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

describe("Reset expert", () => {
  test("asks first, saying what is erased, archived and kept", async () => {
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
    expect(dialog.textContent).toContain("can't be undone");
    expect(dialog.textContent).toContain("Everything Maria remembers");
    expect(dialog.textContent).toContain("Maria's computer and the notes on it");
    expect(dialog.textContent).toContain("leave your chat list");
    expect(dialog.textContent).toContain("what the rest of your team knows");
    expect(reset).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "Reset Maria?" })).toBeNull(),
    );
    expect(reset).not.toHaveBeenCalled();
  });

  test("resets on confirm and lets Maria introduce themselves again", async () => {
    server.use(
      getResetExpertMockHandler({
        memory_nodes_erased: 3,
        computer_erased: true,
        soul_reset: true,
        routines_reset: 1,
        chats_archived: 2,
      }),
    );
    const user = userEvent.setup();
    render(<ExpertResetSection expert={maria} />);

    await user.click(screen.getByTestId("expert-reset-button"));
    await user.click(await screen.findByTestId("reset-expert-confirm"));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Maria is reset" }),
      ),
    );
    expect(
      window.localStorage.getItem(kickoffStorageKey("user-1", "expert-maria")),
    ).toBeNull();
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "Reset Maria?" })).toBeNull(),
    );
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
