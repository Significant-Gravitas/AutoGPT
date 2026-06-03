import {
  getGetSearchGlobalSearchMockHandler,
  getGetSearchGlobalSearchMockHandler200,
} from "@/app/api/__generated__/endpoints/search/search.msw";
import type { GlobalSearchResponse } from "@/app/api/__generated__/models/globalSearchResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GlobalSearchModal } from "../GlobalSearchModal";
import { NAV_BUCKET_LABEL } from "../navigation";

function fixedResponse(
  overrides: Partial<GlobalSearchResponse> = {},
): GlobalSearchResponse {
  return {
    agents: [
      {
        id: "a1",
        type: "library_agent",
        title: "Alpha agent",
        subtitle: "library row",
        score: 0.9,
        updated_at: new Date("2025-01-01T00:00:00Z"),
      },
      {
        id: "a2",
        type: "store_agent",
        title: "Bravo store",
        subtitle: "by acme",
        score: 0.8,
        updated_at: new Date("2025-01-01T00:00:00Z"),
      },
    ],
    files: [
      {
        id: "f1",
        type: "workspace_file",
        title: "notes.md",
        subtitle: "/docs",
        score: 0.7,
        updated_at: new Date("2025-01-01T00:00:00Z"),
      },
    ],
    chats: [
      {
        id: "c1",
        type: "chat_session",
        title: "Charlie chat",
        score: 0.6,
        updated_at: new Date("2025-01-01T00:00:00Z"),
      },
    ],
    ...overrides,
  };
}

function renderModal(
  props: {
    isOpen?: boolean;
    onClose?: () => void;
    onSelectItem?: (item: { id: string }) => void;
  } = {},
) {
  const onClose = props.onClose ?? vi.fn();
  const onSelectItem = props.onSelectItem ?? vi.fn();
  const utils = render(
    <GlobalSearchModal
      isOpen={props.isOpen ?? true}
      onClose={onClose}
      onSelectItem={onSelectItem}
    />,
  );
  return { ...utils, onClose, onSelectItem };
}

describe("GlobalSearchModal", () => {
  it("returns null when closed", () => {
    renderModal({ isOpen: false });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("auto-focuses the search input when opened", async () => {
    server.use(getGetSearchGlobalSearchMockHandler(fixedResponse()));
    renderModal();
    const input = await screen.findByRole("textbox", {
      name: /global search/i,
    });
    await vi.waitFor(() => expect(document.activeElement).toBe(input));
  });

  it("renders bucketed results from the API", async () => {
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    renderModal();

    expect(await screen.findByText("Alpha agent")).toBeDefined();
    expect(screen.getByText("Bravo store")).toBeDefined();
    expect(screen.getByText("notes.md")).toBeDefined();
    expect(screen.getByText("Charlie chat")).toBeDefined();

    // Section headers are rendered for each non-empty bucket.
    expect(screen.getByText("Agents")).toBeDefined();
    expect(screen.getByText("Files")).toBeDefined();
    expect(screen.getByText("Chats")).toBeDefined();
  });

  it("hides empty buckets", async () => {
    server.use(
      getGetSearchGlobalSearchMockHandler200(
        fixedResponse({ files: [], chats: [] }),
      ),
    );
    renderModal();

    expect(await screen.findByText("Alpha agent")).toBeDefined();
    expect(screen.queryByText("Files")).toBeNull();
    expect(screen.queryByText("Chats")).toBeNull();
  });

  it("shows navigation destinations when the API returns no recent items", async () => {
    server.use(
      getGetSearchGlobalSearchMockHandler200({
        agents: [],
        files: [],
        chats: [],
      }),
    );
    renderModal();

    // Navigation is always available, so the idle empty-state never
    // shows; the navigation destinations stand in for an empty list.
    // Scope to the result list so the "Navigate" footer hint (the
    // ``↑↓ Navigate`` shortcut) doesn't collide with the section label.
    const results = await screen.findByRole("listbox", {
      name: "Search results",
    });
    expect(within(results).getByText(NAV_BUCKET_LABEL)).toBeDefined();
    expect(screen.getByRole("option", { name: /builder/i })).toBeDefined();
    expect(screen.queryByText("No recent items")).toBeNull();
  });

  it("shows a query-specific empty state when searching with no results", async () => {
    const user = userEvent.setup();
    server.use(
      getGetSearchGlobalSearchMockHandler200({
        agents: [],
        files: [],
        chats: [],
      }),
    );
    renderModal();

    const input = await screen.findByRole("textbox", {
      name: /global search/i,
    });
    await user.type(input, "zzzz");
    expect(await screen.findByText("No results found")).toBeDefined();
  });

  it("invokes onSelectItem and onClose when a result is clicked", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    const onClose = vi.fn();
    const onSelectItem = vi.fn();
    renderModal({ onClose, onSelectItem });

    const result = await screen.findByRole("option", { name: /alpha agent/i });
    await user.click(result);

    expect(onSelectItem).toHaveBeenCalledTimes(1);
    expect(onSelectItem.mock.calls[0][0]).toMatchObject({
      id: "a1",
      type: "library_agent",
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("navigates results with arrow keys and selects with Enter across buckets", async () => {
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    const onSelectItem = vi.fn();
    renderModal({ onSelectItem });

    // Wait for results to be present.
    await screen.findByText("Charlie chat");
    const dialog = screen.getByRole("dialog");

    // Bucket order is Chats → Agents → Files; first chat is highlighted
    // by default.
    const charlie = screen.getByRole("option", { name: /charlie chat/i });
    expect(charlie.getAttribute("aria-selected")).toBe("true");

    // Three arrow-downs walks chat → agent-1 → agent-2 → files (notes.md).
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    await vi.waitFor(() => {
      const notes = screen.getByRole("option", { name: /notes\.md/i });
      expect(notes.getAttribute("aria-selected")).toBe("true");
    });

    // Enter selects the currently highlighted item.
    fireEvent.keyDown(dialog, { key: "Enter" });
    expect(onSelectItem).toHaveBeenCalledTimes(1);
    expect(onSelectItem.mock.calls[0][0]).toMatchObject({
      id: "f1",
      type: "workspace_file",
    });
  });

  it("closes when Escape is pressed", async () => {
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    const onClose = vi.fn();
    renderModal({ onClose });

    await screen.findByText("Alpha agent");
    const dialog = screen.getByRole("dialog");
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("clears the query when the clear button is pressed", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    renderModal();

    const input = (await screen.findByRole("textbox", {
      name: /global search/i,
    })) as HTMLInputElement;

    await user.type(input, "alpha");
    expect(input.value).toBe("alpha");

    const clearButton = await screen.findByRole("button", {
      name: /clear search/i,
    });
    await user.click(clearButton);
    expect(input.value).toBe("");
  });

  it("shows navigation matches below search results while typing", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    renderModal();

    await screen.findByText("Alpha agent");
    const input = screen.getByRole("textbox", { name: /global search/i });
    await user.type(input, "build");

    // Navigation section surfaces the Builder destination. Scope to the
    // result list so the "Navigate" footer hint doesn't collide.
    const navResults = await screen.findByRole("listbox", {
      name: "Search results",
    });
    expect(within(navResults).getByText(NAV_BUCKET_LABEL)).toBeDefined();
    // The matched substring ("Build") renders in its own span, so the
    // option's accessible name is "Build er" — allow the split.
    const builder = screen.getByRole("option", { name: /build\s*er/i });
    const alpha = screen.getByRole("option", { name: /alpha agent/i });

    // Non-exact match keeps navigation after the search results.
    expect(
      builder.compareDocumentPosition(alpha) & Node.DOCUMENT_POSITION_PRECEDING,
    ).toBeTruthy();
  });

  it("hoists navigation above search results on an exact name match", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    renderModal();

    await screen.findByText("Alpha agent");
    const input = screen.getByRole("textbox", { name: /global search/i });
    await user.type(input, "Builder");

    const builder = await screen.findByRole("option", { name: /builder/i });
    const alpha = screen.getByRole("option", { name: /alpha agent/i });

    // Exact match: Builder is rendered before the first search result.
    expect(
      builder.compareDocumentPosition(alpha) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("shows a spinner and stays open while a navigation item is routing", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    const onClose = vi.fn();
    const onSelectItem = vi.fn();
    renderModal({ onClose, onSelectItem });

    const input = await screen.findByRole("textbox", {
      name: /global search/i,
    });
    await user.type(input, "Builder");

    const builder = await screen.findByRole("option", { name: /builder/i });
    await user.click(builder);

    // Navigation routes internally via the router, not through the
    // ``onSelectItem`` API-item contract. The modal stays open with a
    // trailing spinner until the destination page mounts.
    expect(onSelectItem).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(await screen.findByLabelText("Opening")).toBeDefined();
  });

  it("exposes the Copy user ID action and closes when chosen", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    const onClose = vi.fn();
    const onSelectItem = vi.fn();
    renderModal({ onClose, onSelectItem });

    const input = await screen.findByRole("textbox", {
      name: /global search/i,
    });
    await user.type(input, "copy");

    expect(await screen.findByText("Actions")).toBeDefined();
    const action = await screen.findByRole("option", {
      name: /copy user id/i,
    });
    await user.click(action);

    // Actions run a side effect locally, never the API-item contract,
    // then close once the work settles.
    expect(onSelectItem).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it("highlights the matching query substring inside the title", async () => {
    const user = userEvent.setup();
    server.use(getGetSearchGlobalSearchMockHandler200(fixedResponse()));
    renderModal();

    await screen.findByText("Alpha agent");
    const input = screen.getByRole("textbox", { name: /global search/i });
    await user.type(input, "alpha");

    // After debounce, the matching substring is rendered as a bold span.
    const bolded = await screen.findByText(
      (_content, element) =>
        !!element &&
        element.tagName === "SPAN" &&
        element.className.includes("font-semibold") &&
        element.textContent?.toLocaleLowerCase() === "alpha",
    );
    expect(bolded).toBeDefined();
  });
});
