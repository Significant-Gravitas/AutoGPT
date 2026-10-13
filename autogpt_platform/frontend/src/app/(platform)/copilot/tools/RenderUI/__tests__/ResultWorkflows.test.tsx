import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";

function submitted(send: ReturnType<typeof vi.fn>) {
  return JSON.parse(
    String(send.mock.calls.at(-1)?.[0]).split("Submitted values:\n")[1],
  );
}
describe("result choices and artifacts in chat", () => {
  beforeEach(() => sessionStorage.clear());
  it("lets a user choose among five real options, undo, and retain an unknown price", async () => {
    const options = Array.from({ length: 5 }, (_, index) => ({
      id: `hotel${index}`,
      title: `Hotel ${index}`,
      description: "Supplied option",
      facts: [],
      amount: index ? index * 50 : null,
    }));
    const send = vi.fn();
    await showResult(
      send,
      false,
      formView(
        `ResultList("hotel", "Hotels", ${JSON.stringify(options)}, [], "single", true, "USD/night")`,
      ),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Hotel 0" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Choose Hotel 4" }));
    fireEvent.click(screen.getByRole("button", { name: /Undo/ }));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(submitted(send).hotel).toEqual(["hotel0"]);
    expect(submitted(send).hotel_amount).toBeNull();
  });
  it("shows generated workspace images inline and exposes recoverable loading errors", async () => {
    await showResult(
      vi.fn(),
      false,
      'root = Workspace("Image result", "", [Artifact("Generated cover", "workspace://generated-file#image/png", "image/png", "cover.png", "A blue mountain illustration")])',
    );
    const image = await screen.findByRole("img", {
      name: "A blue mountain illustration",
    });
    expect(image.getAttribute("src")).toBe(
      "/api/proxy/api/workspace/files/generated-file/download",
    );
    fireEvent.error(image);
    expect(screen.getByText(/This file could not load/)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Retry preview" }));
    expect(
      screen.getByRole("img", { name: "A blue mountain illustration" }),
    ).toBeDefined();
  });
  it("previews selectable workspace images with retry and protects private images on shares", async () => {
    const options = [
      {
        id: "cover",
        title: "Cover",
        description: "Uploaded image option",
        facts: [],
        image: { url: "workspace://cover-file#image/png", alt: "Blue cover" },
      },
    ];
    const source = formView(
      `ResultList("cover", "Choose a cover", ${JSON.stringify(options)}, [], "single", false, "")`,
    );
    const view = await showResult(vi.fn(), false, source);
    const image = await screen.findByRole("img", { name: "Blue cover" });
    expect(image.getAttribute("src")).toBe(
      "/api/proxy/api/workspace/files/cover-file/download",
    );
    fireEvent.error(image);
    fireEvent.click(screen.getByRole("button", { name: "Retry Blue cover" }));
    expect(screen.getByRole("img", { name: "Blue cover" })).toBeDefined();
    view.unmount();
    await showResult(vi.fn(), true, source);
    expect(screen.queryByRole("img", { name: "Blue cover" })).toBeNull();
    expect(screen.getByText(/image is private/)).toBeDefined();
  });
});
