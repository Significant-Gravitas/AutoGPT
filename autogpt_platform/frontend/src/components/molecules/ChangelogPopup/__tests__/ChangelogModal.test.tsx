import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChangelogModal } from "../components/ChangelogModal";
import type { ChangelogEntry } from "../helpers";

const ENTRIES: ChangelogEntry[] = [
  {
    slug: "may-7-june-10-2026",
    dateRange: "May 7 – June 10, 2026",
    highlights: "Copilot upgrades",
    url: "https://agpt.co/docs/platform/changelog/changelog/may-7-june-10-2026",
  },
  {
    slug: "april-10-may-1-2026",
    dateRange: "Apr 1 – May 6, 2026",
    highlights: "Marketplace redesign",
    url: "https://agpt.co/docs/platform/changelog/changelog/april-10-may-1-2026",
  },
];

describe("ChangelogModal", () => {
  it("lists every release, each linking out to its docs page", () => {
    render(<ChangelogModal entries={ENTRIES} onClose={vi.fn()} />);

    const first = screen.getByRole("link", { name: /Copilot upgrades/i });
    expect(first.getAttribute("href")).toBe(ENTRIES[0].url);
    expect(first.getAttribute("target")).toBe("_blank");

    expect(
      screen.getByRole("link", { name: /Marketplace redesign/i }),
    ).toBeDefined();
    expect(screen.getByText("May 7 – June 10, 2026")).toBeDefined();
  });

  it("has a 'View all on docs' link", () => {
    render(<ChangelogModal entries={ENTRIES} onClose={vi.fn()} />);
    const viewAll = screen.getByRole("link", { name: /view all on docs/i });
    expect(viewAll.getAttribute("href")).toContain(
      "/docs/platform/changelog/changelog",
    );
  });

  it("fires onClose from the close button", async () => {
    const onClose = vi.fn();
    render(<ChangelogModal entries={ENTRIES} onClose={onClose} />);

    await userEvent.click(screen.getByLabelText("Close changelog"));
    expect(onClose).toHaveBeenCalled();
  });
});
