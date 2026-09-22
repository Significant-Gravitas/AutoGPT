import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { GlobalSearchOverlay } from "../GlobalSearchOverlay";
import { useGlobalSearchStore } from "../useGlobalSearchStore";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
  };
});

describe("GlobalSearchOverlay — flag disabled", () => {
  it("renders nothing and ignores Cmd+K when the flag is off", () => {
    useGlobalSearchStore.setState({ isOpen: false });
    render(<GlobalSearchOverlay />);

    fireEvent.keyDown(document, { key: "k", metaKey: true });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(useGlobalSearchStore.getState().isOpen).toBe(false);
  });
});
