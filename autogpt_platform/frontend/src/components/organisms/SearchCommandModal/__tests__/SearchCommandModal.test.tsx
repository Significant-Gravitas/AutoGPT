import { BookOpenIcon, ChatCircleIcon } from "@phosphor-icons/react";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { SearchCommandBucket } from "../helpers";
import { SearchCommandModal } from "../SearchCommandModal";

const BASE_BUCKETS: SearchCommandBucket[] = [
  {
    key: "agents",
    label: "Agents",
    items: [
      {
        id: "a1",
        title: "Alpha agent",
        subtitle: "library row",
        icon: BookOpenIcon,
      },
      {
        id: "a2",
        title: "Bravo store",
        subtitle: "by acme",
        icon: BookOpenIcon,
      },
    ],
  },
  {
    key: "files",
    label: "Files",
    items: [
      {
        id: "f1",
        title: "notes.md",
        subtitle: "/docs",
        icon: BookOpenIcon,
      },
    ],
  },
  {
    key: "chats",
    label: "Chats",
    items: [
      {
        id: "c1",
        title: "Charlie chat",
        icon: ChatCircleIcon,
      },
    ],
  },
];

interface HarnessProps {
  isOpen?: boolean;
  onClose?: () => void;
  buckets?: SearchCommandBucket[];
  onSelectItem?: Parameters<typeof SearchCommandModal>[0]["onSelectItem"];
  isLoading?: boolean;
  isError?: boolean;
  initialQuery?: string;
}

function Harness({
  isOpen = true,
  onClose = () => {},
  buckets = BASE_BUCKETS,
  onSelectItem = () => {},
  isLoading,
  isError,
  initialQuery = "",
}: HarnessProps) {
  const [query, setQuery] = useState(initialQuery);
  return (
    <SearchCommandModal
      isOpen={isOpen}
      onClose={onClose}
      query={query}
      onQueryChange={setQuery}
      buckets={buckets}
      isLoading={isLoading}
      isError={isError}
      onSelectItem={onSelectItem}
      inputAriaLabel="Test search"
      placeholder="Search…"
      idleEmptyLabel="No items"
      searchingEmptyLabel="No results"
    />
  );
}

describe("SearchCommandModal", () => {
  it("returns null when closed", () => {
    render(<Harness isOpen={false} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("auto-focuses the search input when opened", async () => {
    render(<Harness />);
    const input = await screen.findByRole("textbox", { name: /test search/i });
    await vi.waitFor(() => expect(document.activeElement).toBe(input));
  });

  it("renders bucketed results with section headers", () => {
    render(<Harness />);
    expect(screen.getByText("Alpha agent")).toBeDefined();
    expect(screen.getByText("Bravo store")).toBeDefined();
    expect(screen.getByText("notes.md")).toBeDefined();
    expect(screen.getByText("Charlie chat")).toBeDefined();
    expect(screen.getByText("Agents")).toBeDefined();
    expect(screen.getByText("Files")).toBeDefined();
    expect(screen.getByText("Chats")).toBeDefined();
  });

  it("hides empty buckets", () => {
    render(
      <Harness
        buckets={[
          BASE_BUCKETS[0],
          { ...BASE_BUCKETS[1], items: [] },
          { ...BASE_BUCKETS[2], items: [] },
        ]}
      />,
    );
    expect(screen.queryByText("Files")).toBeNull();
    expect(screen.queryByText("Chats")).toBeNull();
  });

  it("shows the idle empty label with no items and no query", () => {
    render(<Harness buckets={[]} />);
    expect(screen.getByText("No items")).toBeDefined();
  });

  it("shows the searching empty label when a query is active and no items match", () => {
    render(<Harness buckets={[]} initialQuery="zzz" />);
    expect(screen.getByText("No results")).toBeDefined();
  });

  it("renders the loading skeleton", () => {
    render(<Harness buckets={[]} isLoading />);
    expect(screen.getByTestId("search-command-skeleton")).toBeDefined();
  });

  it("renders the error label", () => {
    render(<Harness buckets={[]} isError />);
    expect(screen.getByText("Something went wrong. Try again.")).toBeDefined();
  });

  it("invokes onSelectItem when a result is clicked", async () => {
    const user = userEvent.setup();
    const onSelectItem = vi.fn();
    render(<Harness onSelectItem={onSelectItem} />);

    await user.click(screen.getByRole("option", { name: /alpha agent/i }));
    expect(onSelectItem).toHaveBeenCalledTimes(1);
    expect(onSelectItem.mock.calls[0][0]).toMatchObject({ id: "a1" });
    expect(onSelectItem.mock.calls[0][1]).toBe("agents");
  });

  it("navigates with arrow keys and selects with Enter across buckets", async () => {
    const onSelectItem = vi.fn();
    render(<Harness onSelectItem={onSelectItem} />);
    const dialog = screen.getByRole("dialog");

    expect(
      screen
        .getByRole("option", { name: /alpha agent/i })
        .getAttribute("aria-selected"),
    ).toBe("true");

    // Two arrow-downs walk past both Agents items into Files.
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    expect(
      screen
        .getByRole("option", { name: /notes\.md/i })
        .getAttribute("aria-selected"),
    ).toBe("true");

    fireEvent.keyDown(dialog, { key: "Enter" });
    expect(onSelectItem).toHaveBeenCalledTimes(1);
    expect(onSelectItem.mock.calls[0][0]).toMatchObject({ id: "f1" });
    expect(onSelectItem.mock.calls[0][1]).toBe("files");
  });

  it("clamps arrow navigation at the bounds", () => {
    render(<Harness />);
    const dialog = screen.getByRole("dialog");

    // ArrowUp at the top stays on the first item.
    fireEvent.keyDown(dialog, { key: "ArrowUp" });
    expect(
      screen
        .getByRole("option", { name: /alpha agent/i })
        .getAttribute("aria-selected"),
    ).toBe("true");

    // Press ArrowDown more times than there are items.
    for (let i = 0; i < 10; i += 1) {
      fireEvent.keyDown(dialog, { key: "ArrowDown" });
    }
    expect(
      screen
        .getByRole("option", { name: /charlie chat/i })
        .getAttribute("aria-selected"),
    ).toBe("true");
  });

  it("closes when Escape is pressed", () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    // Radix listens at document level for Escape inside its Dialog;
    // a keyDown on the document mirrors what a user keystroke would.
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not close when the dialog body itself is clicked", () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    fireEvent.mouseDown(screen.getByRole("dialog"));
    expect(onClose).not.toHaveBeenCalled();
  });

  it("clears the query when the clear button is pressed", async () => {
    const user = userEvent.setup();
    render(<Harness initialQuery="alpha" />);

    const input = screen.getByRole("textbox", {
      name: /test search/i,
    }) as HTMLInputElement;
    expect(input.value).toBe("alpha");

    await user.click(screen.getByRole("button", { name: /clear search/i }));
    expect(input.value).toBe("");
  });

  it("highlights the matching query substring inside the title", () => {
    render(<Harness initialQuery="alpha" />);

    const bolded = screen.getByText(
      (_content, element) =>
        !!element &&
        element.tagName === "SPAN" &&
        element.className.includes("font-semibold") &&
        element.textContent?.toLocaleLowerCase() === "alpha",
    );
    expect(bolded).toBeDefined();
  });
});
