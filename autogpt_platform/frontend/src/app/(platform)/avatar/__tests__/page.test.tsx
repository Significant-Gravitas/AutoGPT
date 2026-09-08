import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import AvatarPage from "../page";

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

const toastMock = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

function stageAvatar() {
  return within(screen.getByTestId("avatar-stage")).getByTestId("bot-avatar");
}

function pickerRadio(group: string, name: string) {
  return within(screen.getByRole("radiogroup", { name: group })).getByRole(
    "radio",
    { name },
  );
}

afterEach(() => {
  vi.restoreAllMocks();
  toastMock.mockClear();
});

describe("AvatarPage", () => {
  test("starts from the default expert and updates the stage per pick", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    expect(stageAvatar().getAttribute("data-avatar")).toBe(
      "round.lavender.none",
    );

    await user.click(pickerRadio("Shape", "Egg"));
    await user.click(pickerRadio("Colour", "Mint"));
    await user.click(pickerRadio("Accessory", "Headset"));

    expect(stageAvatar().getAttribute("data-avatar")).toBe("egg.mint.headset");
    expect(pickerRadio("Shape", "Egg").getAttribute("aria-checked")).toBe(
      "true",
    );
    expect(pickerRadio("Shape", "Round").getAttribute("aria-checked")).toBe(
      "false",
    );
  });

  test("status toggle drives the expression without touching the config", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    await user.click(pickerRadio("Status", "Needs you"));
    expect(stageAvatar().getAttribute("data-status")).toBe("waiting");
    expect(screen.getByText("blocked, asking")).toBeDefined();

    await user.click(pickerRadio("Status", "Done"));
    expect(stageAvatar().getAttribute("data-status")).toBe("done");
    expect(stageAvatar().getAttribute("data-avatar")).toBe(
      "round.lavender.none",
    );
  });

  test("shuffle changes the config and reset restores the default", async () => {
    const user = userEvent.setup();
    vi.spyOn(Math, "random").mockReturnValue(0.99);
    render(<AvatarPage />);

    await user.click(screen.getByRole("button", { name: "Shuffle" }));
    expect(stageAvatar().getAttribute("data-avatar")).toBe(
      "cloud.butter.headband",
    );

    await user.click(pickerRadio("Status", "Working"));
    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(stageAvatar().getAttribute("data-avatar")).toBe(
      "round.lavender.none",
    );
    expect(stageAvatar().getAttribute("data-status")).toBe("idle");
  });

  test("turn sliders pose the stage avatar and reset clears them", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    const yaw = screen.getByRole("slider", { name: "Turn" });
    fireEvent.change(yaw, { target: { value: "40" } });
    expect((yaw as HTMLInputElement).value).toBe("40");
    expect(screen.getByText("40°")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect((yaw as HTMLInputElement).value).toBe("0");
  });

  test("outline switch flips every avatar on the page", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    expect(stageAvatar().getAttribute("data-outline")).toBe("false");
    await user.click(screen.getByRole("switch", { name: "Outline" }));
    expect(stageAvatar().getAttribute("data-outline")).toBe("true");
    const avatars = screen.getAllByTestId("bot-avatar");
    expect(
      avatars.every((svg) => svg.getAttribute("data-outline") === "true"),
    ).toBe(true);
  });

  test("pinning an expression overrides the status pool until reset", async () => {
    const user = userEvent.setup();
    render(<AvatarPage />);

    expect(stageAvatar().getAttribute("data-expression")).toBe("neutral");
    await user.click(pickerRadio("Expression", "Angry"));
    expect(stageAvatar().getAttribute("data-expression")).toBe("angry");

    await user.click(pickerRadio("Status", "Done"));
    expect(stageAvatar().getAttribute("data-expression")).toBe("angry");

    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(stageAvatar().getAttribute("data-expression")).toBe("neutral");
  });

  test("copy link writes the current url and confirms", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    render(<AvatarPage />);

    await user.click(screen.getByRole("button", { name: "Copy link" }));
    expect(writeText).toHaveBeenCalledWith(window.location.href);
    expect(toastMock).toHaveBeenCalledWith({ title: "Link copied" });
  });

  test("roster preview lists your expert first alongside sample teammates", () => {
    render(<AvatarPage />);
    const rows = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(rows).toHaveLength(7);
    expect(within(rows[0]).getByText("Yours")).toBeDefined();
    expect(
      within(rows[0]).getByTestId("bot-avatar").getAttribute("data-avatar"),
    ).toBe("round.lavender.none");
  });
});
