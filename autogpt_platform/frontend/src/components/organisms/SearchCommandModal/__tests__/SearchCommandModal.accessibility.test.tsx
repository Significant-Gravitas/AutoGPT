import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SearchCommandModal } from "../SearchCommandModal";

describe("SearchCommandModal accessibility", () => {
  it("describes the dialog content for assistive technology", () => {
    render(
      <SearchCommandModal
        isOpen
        onClose={vi.fn()}
        query=""
        onQueryChange={vi.fn()}
        buckets={[]}
        onSelectItem={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "Search" });
    const descriptionId = dialog.getAttribute("aria-describedby");

    expect(descriptionId).toBeTruthy();
    expect(document.getElementById(descriptionId ?? "")?.textContent).toBe(
      "Search commands and results.",
    );
  });
});
