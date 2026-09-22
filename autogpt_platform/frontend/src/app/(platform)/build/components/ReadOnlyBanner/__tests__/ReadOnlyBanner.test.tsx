import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockDuplicate = vi.fn();
let mockCanDuplicate = true;
let mockIsDuplicating = false;
let mockIsCheckingLibrary = false;

vi.mock("../../../hooks/useDuplicateGraph", () => ({
  useDuplicateGraph: () => ({
    duplicate: mockDuplicate,
    isDuplicating: mockIsDuplicating,
    canDuplicate: mockCanDuplicate,
    isCheckingLibrary: mockIsCheckingLibrary,
  }),
}));

import { ReadOnlyBanner } from "../ReadOnlyBanner";

describe("ReadOnlyBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCanDuplicate = true;
    mockIsDuplicating = false;
    mockIsCheckingLibrary = false;
  });

  test("shows the Duplicate CTA when the agent can be duplicated", () => {
    render(<ReadOnlyBanner />);

    const button = screen.getByRole("button", { name: /duplicate/i });
    expect(button).not.toBeNull();
    expect(screen.getByText(/Duplicate it to make changes/i)).not.toBeNull();

    fireEvent.click(button);
    expect(mockDuplicate).toHaveBeenCalledTimes(1);
  });

  test("explains how to enable duplication and hides the CTA when it cannot duplicate", () => {
    mockCanDuplicate = false;
    render(<ReadOnlyBanner />);

    expect(screen.queryByRole("button", { name: /duplicate/i })).toBeNull();
    expect(
      screen.getByText(/Add it to your library to enable duplication/i),
    ).not.toBeNull();
  });

  test("shows the Duplicate CTA disabled while the library lookup is in flight", () => {
    mockCanDuplicate = false;
    mockIsCheckingLibrary = true;
    render(<ReadOnlyBanner />);

    const button = screen.getByRole("button", {
      name: /duplicate/i,
    }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(screen.getByText(/Duplicate it to make changes/i)).not.toBeNull();
    expect(
      screen.queryByText(/Add it to your library to enable duplication/i),
    ).toBeNull();
  });

  test("announces the read-only state to assistive technology", () => {
    const { container } = render(<ReadOnlyBanner />);

    const banner = container.querySelector('[data-id="read-only-banner"]');
    expect(banner?.getAttribute("role")).toBe("status");
    expect(banner?.getAttribute("aria-live")).toBe("polite");
  });

  test("can be dismissed", () => {
    const { container } = render(<ReadOnlyBanner />);

    expect(
      container.querySelector('[data-id="read-only-banner"]'),
    ).not.toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));

    expect(container.querySelector('[data-id="read-only-banner"]')).toBeNull();
  });
});
