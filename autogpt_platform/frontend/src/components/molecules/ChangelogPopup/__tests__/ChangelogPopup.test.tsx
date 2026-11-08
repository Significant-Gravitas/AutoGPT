import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { ChangelogPopup } from "../ChangelogPopup";

const LATEST = "june-18-july-9-2026";
const INDEX_MD = `
| Date | Highlights |
| ---- | ---------- |
| [June 18 – July 9](/docs/platform/changelog/changelog/${LATEST}.md) | File folders, shareable chats, Discord bot |
`;

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
  );
});

afterEach(() => {
  server.resetHandlers();
  vi.useRealTimers();
});

describe("ChangelogPopup", () => {
  it("slides in for an unseen release with highlights and a View changelog link", async () => {
    render(<ChangelogPopup />);

    expect(
      await screen.findByText("File folders", {}, { timeout: 3000 }),
    ).toBeDefined();
    expect(screen.getByText("Shareable chats")).toBeDefined();
    expect(screen.getByText("Discord bot")).toBeDefined();

    const link = screen.getByRole("link", { name: /view changelog/i });
    expect(link.getAttribute("href")).toContain(`/changelog/${LATEST}`);
    expect(link.getAttribute("target")).toBe("_blank");
  });

  it("can be dismissed", async () => {
    render(<ChangelogPopup />);

    const dismiss = await screen.findByLabelText(
      "Dismiss changelog",
      {},
      { timeout: 3000 },
    );
    await userEvent.click(dismiss);

    await waitFor(() => expect(screen.queryByText("File folders")).toBeNull());
  });

  it("stays hidden for a release the user has already seen", async () => {
    window.localStorage.setItem(STORAGE_KEY, LATEST);
    render(<ChangelogPopup />);

    await new Promise((resolve) => setTimeout(resolve, 1800));
    expect(screen.queryByText("File folders")).toBeNull();
  });

  it("dismisses when the release link is activated while hovered and focused", async () => {
    vi.useFakeTimers();
    render(<ChangelogPopup />);
    await vi.waitFor(() => expect(screen.getByRole("dialog")).toBeDefined(), {
      timeout: 3000,
    });
    const popup = screen.getByRole("dialog");
    const link = screen.getByRole("link", { name: /view changelog/i });
    fireEvent.pointerEnter(popup, { pointerType: "mouse" });
    fireEvent.mouseEnter(popup);
    fireEvent.focus(link);
    link.addEventListener("click", (event) => event.preventDefault(), {
      once: true,
    });
    fireEvent.click(link);

    await act(() => vi.advanceTimersByTimeAsync(500));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
  });

  it("keeps the countdown after a touch tap synthesizes mouse events", async () => {
    vi.useFakeTimers();
    render(<ChangelogPopup />);
    await vi.waitFor(() => expect(screen.getByRole("dialog")).toBeDefined(), {
      timeout: 3000,
    });
    await act(() => vi.advanceTimersByTimeAsync(4000));
    const popup = screen.getByRole("dialog");
    fireEvent.pointerEnter(popup, { pointerType: "touch" });
    fireEvent.pointerDown(popup, { pointerType: "touch" });
    fireEvent.pointerUp(popup, { pointerType: "touch" });
    fireEvent.pointerLeave(popup, { pointerType: "touch" });
    fireEvent.mouseEnter(popup);
    fireEvent.click(popup);

    await act(() => vi.advanceTimersByTimeAsync(4500));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("resumes the remaining countdown when the hovering pointer is cancelled", async () => {
    vi.useFakeTimers();
    render(<ChangelogPopup />);
    await vi.waitFor(() => expect(screen.getByRole("dialog")).toBeDefined(), {
      timeout: 3000,
    });
    await act(() => vi.advanceTimersByTimeAsync(6000));
    const popup = screen.getByRole("dialog");
    fireEvent.pointerEnter(popup, { pointerType: "mouse" });
    fireEvent.mouseEnter(popup);
    await act(() => vi.advanceTimersByTimeAsync(9000));
    expect(screen.getByRole("dialog")).toBeDefined();

    fireEvent.pointerCancel(popup, { pointerType: "mouse" });
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByRole("dialog")).toBeDefined();
    await act(() => vi.advanceTimersByTimeAsync(1500));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it.each(["hover", "focus"])(
    "stays paused until both hover and focus end (%s ends first)",
    async (firstToEnd) => {
      vi.useFakeTimers();
      render(<ChangelogPopup />);
      await vi.waitFor(() => expect(screen.getByRole("dialog")).toBeDefined(), {
        timeout: 3000,
      });
      const popup = screen.getByRole("dialog");
      const link = screen.getByRole("link", { name: /view changelog/i });
      fireEvent.pointerEnter(popup, { pointerType: "mouse" });
      fireEvent.focus(link);

      if (firstToEnd === "hover") {
        fireEvent.pointerLeave(popup, { pointerType: "mouse" });
      } else fireEvent.blur(link, { relatedTarget: document.body });

      await act(() => vi.advanceTimersByTimeAsync(9000));
      expect(screen.getByRole("dialog")).toBeDefined();

      if (firstToEnd === "hover") {
        fireEvent.blur(link, { relatedTarget: document.body });
      } else {
        fireEvent.pointerLeave(popup, { pointerType: "mouse" });
      }

      await act(() => vi.advanceTimersByTimeAsync(8500));
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
    },
  );

  it("keeps the remaining time while focus moves between popup controls", async () => {
    vi.useFakeTimers();
    render(<ChangelogPopup />);
    await vi.waitFor(() => expect(screen.getByRole("dialog")).toBeDefined(), {
      timeout: 3000,
    });
    await act(() => vi.advanceTimersByTimeAsync(6000));

    const dismiss = screen.getByRole("button", { name: "Dismiss changelog" });
    const link = screen.getByRole("link", { name: /view changelog/i });
    fireEvent.focus(dismiss);
    fireEvent.blur(dismiss, { relatedTarget: link });
    fireEvent.focus(link, { relatedTarget: dismiss });
    await act(() => vi.advanceTimersByTimeAsync(9000));
    expect(screen.getByRole("dialog")).toBeDefined();

    fireEvent.blur(link, { relatedTarget: document.body });
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByRole("dialog")).toBeDefined();
    await act(() => vi.advanceTimersByTimeAsync(1500));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
