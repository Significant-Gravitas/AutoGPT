import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChangelogModal } from "../components/ChangelogModal";
import type { ChangelogEntry } from "../helpers";

const ENTRIES: ChangelogEntry[] = [
  {
    slug: "v0-6-63",
    dateRange: "May 7 – June 10, 2026",
    highlights: "Copilot upgrades",
    url: "https://agpt.co/docs/platform/changelog/changelog/v0-6-63",
    mdUrl: "https://agpt.co/docs/platform/changelog/changelog/v0-6-63.md",
  },
  {
    slug: "v0-6-58",
    dateRange: "Apr 1 – May 6, 2026",
    highlights: "Marketplace redesign",
    url: "https://agpt.co/docs/platform/changelog/changelog/v0-6-58",
    mdUrl: "https://agpt.co/docs/platform/changelog/changelog/v0-6-58.md",
  },
];

function renderModal(
  overrides: Partial<Parameters<typeof ChangelogModal>[0]> = {},
) {
  const props = {
    entries: ENTRIES,
    selectedEntry: ENTRIES[0],
    entryMarkdown: null,
    isLoadingMarkdown: false,
    onSelectEntry: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };
  render(<ChangelogModal {...props} />);
  return props;
}

describe("ChangelogModal", () => {
  it("lists every release in the sidebar", () => {
    renderModal();
    expect(screen.getByText("Copilot upgrades")).toBeDefined();
    expect(screen.getByText("Marketplace redesign")).toBeDefined();
  });

  it("renders the selected release's markdown", async () => {
    renderModal({ entryMarkdown: "# Hello changelog" });
    expect(await screen.findByText("Hello changelog")).toBeDefined();
  });

  it("shows the loading state while markdown is fetching", () => {
    renderModal({ isLoadingMarkdown: true });
    expect(screen.getAllByText(/loading/i).length).toBeGreaterThan(0);
  });

  it("shows a fallback when markdown fails to load", () => {
    renderModal({ entryMarkdown: null, isLoadingMarkdown: false });
    expect(screen.getByText(/could not load changelog entry/i)).toBeDefined();
  });

  it("fires onSelectEntry when another release is clicked", async () => {
    const { onSelectEntry } = renderModal();
    await userEvent.click(screen.getByText("Marketplace redesign"));
    expect(onSelectEntry).toHaveBeenCalledWith(ENTRIES[1]);
  });

  it("fires onClose from the close button", async () => {
    const { onClose } = renderModal();
    await userEvent.click(screen.getByLabelText("Close changelog"));
    expect(onClose).toHaveBeenCalled();
  });
});
