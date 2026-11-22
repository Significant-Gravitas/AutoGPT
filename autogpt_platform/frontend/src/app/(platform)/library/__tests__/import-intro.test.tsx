import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import LibraryPage from "../page";

const onboarding = vi.hoisted(() => ({
  ready: false,
  completeStep: vi.fn(),
}));

vi.mock("posthog-js", () => ({ default: { capture: vi.fn() } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "library-intro-user", created_at: "2026-10-01T00:00:00Z" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => ({
    state: onboarding.ready ? { completedSteps: [] } : null,
    completeStep: onboarding.completeStep,
  }),
  default: ({ children }: { children: ReactNode }) => children,
}));

beforeEach(() => {
  window.localStorage.clear();
  onboarding.ready = false;
  onboarding.completeStep.mockReset();
  const pagination = {
    total_items: 0,
    total_pages: 1,
    current_page: 1,
    page_size: 20,
  };
  server.use(
    getGetV2ListLibraryAgentsMockHandler({ agents: [], pagination }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({ agents: [], pagination }),
    getGetV2ListLibraryFoldersMockHandler({ folders: [], pagination }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function prepareDrawer(dialog: HTMLElement) {
  // Happy DOM omits the browser's default computed transform.
  if (dialog.hasAttribute("data-vaul-drawer")) dialog.style.transform = "none";
}

function finishDrawerExit() {
  // Happy DOM does not emit the CSS animation events Radix awaits on close.
  document
    .querySelectorAll(
      '[data-vaul-drawer][data-state="closed"], [data-vaul-overlay][data-state="closed"]',
    )
    .forEach((element) => {
      const animationEnd = new Event("animationend", { bubbles: true });
      Object.defineProperty(animationEnd, "animationName", {
        value: window.getComputedStyle(element).animationName,
      });
      fireEvent(element, animationEnd);
    });
}

test.each(
  [1280, 390].flatMap((width) => [
    { width, closeWithEscape: false },
    { width, closeWithEscape: true },
  ]),
)(
  "defers a late intro until Import closes at width $width (Escape=$closeWithEscape)",
  async ({ width, closeWithEscape }) => {
    vi.stubGlobal("innerWidth", width);
    const { rerender } = render(<LibraryPage />);
    await userEvent.click(screen.getAllByTestId("import-button")[0]);
    const importDialog = await screen.findByRole("dialog", { name: "Import" });
    prepareDrawer(importDialog);

    onboarding.ready = true;
    rerender(<LibraryPage />);
    expect(screen.queryByTestId("tab-intro-overlay")).toBeNull();
    expect(onboarding.completeStep).not.toHaveBeenCalled();
    expect(
      window.localStorage.getItem("autogpt:tab-intro-seen:agents"),
    ).toBeNull();
    expect(screen.getAllByTestId("import-button")).toHaveLength(1);

    await userEvent.click(
      within(importDialog).getByRole("tab", { name: "AutoGPT workflow" }),
    );
    const name = within(importDialog).getByRole("textbox", {
      name: "Workflow name",
    });
    await userEvent.type(name, "Quarterly report");
    act(() => {
      window.innerWidth = width === 1280 ? 390 : 1280;
      window.dispatchEvent(new Event("resize"));
    });
    await waitFor(() =>
      expect(screen.getAllByRole("dialog", { name: "Import" })).toHaveLength(1),
    );
    const resizedDialog = screen.getByRole("dialog", { name: "Import" });
    prepareDrawer(resizedDialog);
    const draftName = within(resizedDialog).getByRole("textbox", {
      name: "Workflow name",
    }) as HTMLInputElement;
    expect(draftName.value).toBe("Quarterly report");
    expect(screen.queryByTestId("tab-intro-overlay")).toBeNull();

    if (closeWithEscape) {
      await userEvent.keyboard("{Escape}");
    } else {
      await userEvent.click(
        within(resizedDialog).getByRole("button", { name: "Close" }),
      );
    }
    if (resizedDialog.hasAttribute("data-vaul-drawer")) {
      expect(resizedDialog.getAttribute("data-state")).toBe("closed");
      finishDrawerExit();
    }
    await waitFor(() => expect(document.contains(resizedDialog)).toBe(false));
    expect(
      await screen.findByRole("dialog", { name: "Your mission control." }),
    ).toBeDefined();
    expect(onboarding.completeStep).not.toHaveBeenCalled();
    expect(
      window.localStorage.getItem("autogpt:tab-intro-seen:agents"),
    ).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));
    await waitFor(() =>
      expect(screen.queryByTestId("tab-intro-overlay")).toBeNull(),
    );
    expect(onboarding.completeStep).toHaveBeenCalledExactlyOnceWith(
      "AGENTS_TAB_INTRO",
    );
    expect(window.localStorage.getItem("autogpt:tab-intro-seen:agents")).toBe(
      "library-intro-user",
    );
  },
);
