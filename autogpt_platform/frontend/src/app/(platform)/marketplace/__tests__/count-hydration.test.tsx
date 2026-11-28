import { render } from "@testing-library/react";
import { renderToString } from "react-dom/server";
import { expect, it, vi } from "vitest";
import { ShelfMoreButton } from "../components/Shelf/ShelfMoreButton";

it("hydrates a marketplace count when the browser and server locales differ", () => {
  const toLocaleString = Number.prototype.toLocaleString;
  let defaultLocale = "en-US";
  const formatter = vi
    .spyOn(Number.prototype, "toLocaleString")
    .mockImplementation(function (this: number, locales, options) {
      return toLocaleString.call(this, locales ?? defaultLocale, options);
    });
  try {
    const ui = (
      <ShelfMoreButton
        isExpanded={false}
        count={1234}
        noun="skills"
        onToggle={() => {}}
      />
    );
    const container = document.createElement("div");
    container.innerHTML = renderToString(ui);
    expect(container.textContent).toBe("Load all 1,234 skills");

    defaultLocale = "de-DE";
    const onRecoverableError = vi.fn();
    render(ui, { container, hydrate: true, onRecoverableError });
    expect(onRecoverableError).not.toHaveBeenCalled();
    expect(container.textContent).toBe("Load all 1,234 skills");
  } finally {
    formatter.mockRestore();
  }
});
