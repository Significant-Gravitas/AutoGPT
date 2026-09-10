import {
  ACCESSORIES,
  configForName,
  parseAvatarUrl,
} from "@/components/molecules/BotAvatar/helpers";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AvatarPage from "../page";

const writeText = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  writeText.mockClear();
});

// userEvent.setup() installs its own clipboard stub, so the spy has to replace
// it afterwards rather than before.
function stubClipboard() {
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
}

function currentUrl() {
  return screen.getByTestId("avatar-url").textContent ?? "";
}

describe("AvatarPage", () => {
  it("starts from a seeded config and shows its url", () => {
    render(<AvatarPage />);

    const seeded = configForName("Otto");
    expect(currentUrl()).toBe(
      `/avatars/${seeded.shape}.${seeded.color}.${seeded.accessory}.svg`,
    );
  });

  it("updates the url when a shape is picked", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByTestId("avatar-option-shape-squircle"));

    expect(parseAvatarUrl(currentUrl())?.shape).toBe("squircle");
  });

  it("updates the url when a colour is picked", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByRole("tab", { name: "Colour" }));
    await user.click(screen.getByTestId("avatar-option-color-mint"));

    expect(parseAvatarUrl(currentUrl())?.color).toBe("mint");
  });

  it("updates the url when an accessory is picked", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByRole("tab", { name: "Accessory" }));
    await user.click(screen.getByTestId("avatar-option-accessory-halo"));

    expect(parseAvatarUrl(currentUrl())?.accessory).toBe("halo");
  });

  it("copies the url to the clipboard", async () => {
    const user = userEvent.setup();
    stubClipboard();
    render(<AvatarPage />);

    const url = currentUrl();
    await user.click(screen.getByRole("button", { name: /copy url/i }));

    expect(writeText).toHaveBeenCalledWith(url);
  });

  it("keeps a valid url after surprise me", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByRole("button", { name: "Surprise me" }));

    expect(parseAvatarUrl(currentUrl())).not.toBeNull();
  });

  it("switches the previewed status", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByRole("button", { name: "Working" }));

    const previews = screen.getAllByTestId("bot-avatar");
    expect(
      previews.some((node) => node.getAttribute("data-status") === "working"),
    ).toBe(true);
  });

  it("switches the previewed expression", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByRole("button", { name: "Suspicious" }));

    const previews = screen.getAllByTestId("bot-avatar");
    expect(
      previews.some(
        (node) => node.getAttribute("data-expression") === "suspicious",
      ),
    ).toBe(true);
  });

  it("resizes the sized preview", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(screen.getByRole("button", { name: "24px" }));

    const previews = screen.getAllByTestId("bot-avatar");
    expect(previews.some((node) => node.getAttribute("width") === "24")).toBe(
      true,
    );
  });

  it("lists every accessory in the review strip", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    const strip = screen
      .getByText("Every accessory on this face")
      .closest("section") as HTMLElement;
    expect(within(strip).getAllByRole("button")).toHaveLength(
      ACCESSORIES.length,
    );

    await user.click(within(strip).getAllByRole("button")[3]);
    expect(parseAvatarUrl(currentUrl())?.accessory).toBe(ACCESSORIES[3].id);
  });
});
