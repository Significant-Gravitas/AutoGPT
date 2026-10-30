import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import MemoryDashboardPage from "../page";

const { renderVisualizer, withRoleAccess } = vi.hoisted(() => ({
  renderVisualizer: vi.fn(),
  withRoleAccess: vi.fn(async () => (component: unknown) => component),
}));

vi.mock("@/lib/withRoleAccess", () => ({ withRoleAccess }));
vi.mock("../components/MemoryVisualizer", () => ({
  MemoryVisualizer: function MemoryVisualizer() {
    renderVisualizer();
    return <div>Memory data</div>;
  },
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllEnvs();
});

describe("Memory Inspector availability", () => {
  it("does not mount memory data in the desktop build", async () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY", "false");
    render(await MemoryDashboardPage());

    expect(
      screen.getByText("Long-term memory is unavailable on this installation."),
    ).toBeDefined();
    expect(renderVisualizer).not.toHaveBeenCalled();
    expect(withRoleAccess).toHaveBeenCalledWith(["admin"]);
  });

  it.each(["CLOUD", "LOCAL"])(
    "preserves the inspector for %s without a desktop override",
    async (mode) => {
      vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", mode);
      vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY", undefined);
      render(await MemoryDashboardPage());

      expect(screen.getByText("Memory data")).toBeDefined();
      expect(withRoleAccess).toHaveBeenCalledWith(["admin"]);
    },
  );
});
