import { getGetSearchGlobalSearchMockHandler200 } from "@/app/api/__generated__/endpoints/search/search.msw";
import type { GlobalSearchResponse } from "@/app/api/__generated__/models/globalSearchResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GlobalSearchOverlay } from "../GlobalSearchOverlay";
import { useGlobalSearchStore } from "../useGlobalSearchStore";

// Override the static next/navigation mock so this file can drive a pathname
// change and assert the palette closes after navigation lands.
let mockPathname = "/marketplace";
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => mockPathname,
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "chat-search",
  };
});

function makeSearchResponse(
  overrides: Partial<GlobalSearchResponse> = {},
): GlobalSearchResponse {
  return {
    agents: [],
    files: [],
    chats: [
      {
        id: "newer",
        type: "chat_session",
        title: "Revenue forecast",
        score: 0.9,
        updated_at: new Date("2025-01-03T00:00:00Z"),
      },
      {
        id: "middle",
        type: "chat_session",
        title: "Forecast follow-up",
        score: 0.8,
        updated_at: new Date("2025-01-02T00:00:00Z"),
      },
      {
        id: "older",
        type: "chat_session",
        title: "Budget notes",
        score: 0.6,
        updated_at: new Date("2025-01-01T00:00:00Z"),
      },
    ],
    ...overrides,
  };
}

describe("GlobalSearchOverlay", () => {
  beforeEach(() => {
    mockPathname = "/marketplace";
    useGlobalSearchStore.setState({ isOpen: false });
    server.use(
      // Search endpoint is filtered server-side by the ``q`` param. To keep
      // the test deterministic we narrow the chat bucket here instead of
      // relying on backend semantics.
      getGetSearchGlobalSearchMockHandler200(({ request }) => {
        const url = new URL(request.url);
        const q = (url.searchParams.get("q") ?? "").trim().toLowerCase();
        const response = makeSearchResponse();
        if (!q) {
          return response;
        }
        response.chats = (response.chats ?? []).filter((chat) =>
          chat.title.toLowerCase().includes(q),
        );
        return response;
      }),
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it("opens with Cmd+K, focuses the input, and shows recent chats", async () => {
    render(<GlobalSearchOverlay />);

    fireEvent.keyDown(document, { key: "k", metaKey: true });

    const dialog = await screen.findByRole("dialog");
    const input = screen.getByRole("textbox", { name: /global search/i });
    await vi.waitFor(() => expect(document.activeElement).toBe(input));
    expect(await within(dialog).findByText("Revenue forecast")).toBeDefined();
  });

  it("filters results, shows empty copy, and clears the query", async () => {
    const user = userEvent.setup();
    render(<GlobalSearchOverlay />);

    fireEvent.keyDown(document, { key: "k", metaKey: true });
    await user.type(
      screen.getByRole("textbox", { name: /global search/i }),
      "forecast",
    );

    const dialog = screen.getByRole("dialog");
    // The hook keeps previous results visible via ``placeholderData`` to
    // avoid a flash-of-empty on every keystroke — wait for the filtered
    // response to land before asserting that ``Budget notes`` is gone.
    await vi.waitFor(() => {
      expect(
        within(dialog).queryByRole("option", { name: /budget notes/i }),
      ).toBeNull();
    });
    expect(
      within(dialog).getByRole("option", { name: /revenue forecast/i }),
    ).toBeDefined();
    expect(
      within(dialog).getByRole("option", { name: /forecast follow-up/i }),
    ).toBeDefined();

    await user.clear(screen.getByRole("textbox", { name: /global search/i }));
    await user.type(
      screen.getByRole("textbox", { name: /global search/i }),
      "missing",
    );
    expect(await screen.findByText("No results found")).toBeDefined();

    await user.click(screen.getByRole("button", { name: /clear search/i }));
    expect(
      (
        screen.getByRole("textbox", {
          name: /global search/i,
        }) as HTMLInputElement
      ).value,
    ).toBe("");
  });

  it("closes the palette after navigation lands on a new route", async () => {
    const { rerender } = render(<GlobalSearchOverlay />);

    fireEvent.keyDown(document, { key: "k", metaKey: true });
    expect(await screen.findByRole("dialog")).toBeDefined();

    mockPathname = "/library";
    rerender(<GlobalSearchOverlay />);

    await vi.waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  it("supports keyboard navigation, Enter selection, and shortcut dismissal", async () => {
    const user = userEvent.setup();
    render(<GlobalSearchOverlay />);

    fireEvent.keyDown(document, { key: "k", metaKey: true });
    const dialog = await screen.findByRole("dialog");

    await user.type(
      screen.getByRole("textbox", { name: /global search/i }),
      "forecast",
    );
    // Wait for the highlighted result to settle on the top match.
    await within(dialog).findByRole("option", { name: /revenue forecast/i });

    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    fireEvent.keyDown(dialog, { key: "Enter" });

    await vi.waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    expect(await screen.findByRole("dialog")).toBeDefined();
    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    await vi.waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });
});
