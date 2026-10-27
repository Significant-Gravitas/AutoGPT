import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ConnectProviderRow } from "../ConnectProviderRow";

const GITHUB = {
  id: "github",
  serviceId: "github",
  name: "GitHub",
  supportedAuthTypes: [],
};

describe("ConnectProviderRow", () => {
  it("falls back to the provider initial when the logo fails to load", async () => {
    render(<ConnectProviderRow provider={GITHUB} onSelect={vi.fn()} />);

    const row = screen.getByRole("button", { name: /GitHub/ });
    const logo = row.querySelector("img");
    expect(logo).not.toBeNull();

    fireEvent.error(logo as HTMLImageElement);

    expect(await within(row).findByText("G")).toBeDefined();
    expect(row.querySelector("img")).toBeNull();
  });

  it("hands back the provider id when picked", async () => {
    const onSelect = vi.fn();
    render(<ConnectProviderRow provider={GITHUB} onSelect={onSelect} />);

    await userEvent.click(screen.getByRole("button", { name: /GitHub/ }));

    expect(onSelect).toHaveBeenCalledWith("github");
  });
});
