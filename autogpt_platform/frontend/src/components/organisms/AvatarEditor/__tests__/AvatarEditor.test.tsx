import {
  configForName,
  DEFAULT_CONFIG,
  isAccessoryId,
  type AvatarConfig,
} from "@/components/molecules/BotAvatar/helpers";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { AvatarEditor } from "../AvatarEditor";
import { FACETS, nextIndex, selectedIndex } from "../helpers";

function Harness({
  onChange,
  name,
}: {
  onChange?: (config: AvatarConfig) => void;
  name?: string;
}) {
  const [config, setConfig] = useState<AvatarConfig>(DEFAULT_CONFIG);
  return (
    <>
      <AvatarEditor
        value={config}
        name={name}
        onChange={(next) => {
          setConfig(next);
          onChange?.(next);
        }}
      />
      <output data-testid="config">{`${config.shape}.${config.color}.${config.accessory}`}</output>
    </>
  );
}

describe("AvatarEditor", () => {
  it("changes the config when an option is clicked", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByTestId("avatar-option-shape-bean"));

    expect(screen.getByTestId("config").textContent).toBe("bean.lavender.none");
  });

  it("moves through options with the arrow keys", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const first = screen.getByTestId("avatar-option-shape-round");
    first.focus();
    await user.keyboard("{ArrowRight}");

    expect(screen.getByTestId("config").textContent).toBe("dome.lavender.none");

    await user.keyboard("{ArrowLeft}");
    expect(screen.getByTestId("config").textContent).toBe(
      "round.lavender.none",
    );
  });

  it("marks the selected option for assistive tech", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByTestId("avatar-option-shape-wide"));

    expect(
      screen
        .getByTestId("avatar-option-shape-wide")
        .getAttribute("aria-checked"),
    ).toBe("true");
    expect(
      screen
        .getByTestId("avatar-option-shape-round")
        .getAttribute("aria-checked"),
    ).toBe("false");
  });

  it("switches facets and picks an accessory", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("tab", { name: "Accessory" }));
    await user.click(screen.getByTestId("avatar-option-accessory-tophat"));

    expect(screen.getByTestId("config").textContent).toBe(
      "round.lavender.tophat",
    );
  });

  it("seeds a deterministic config from a name", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(screen.getByLabelText("Seed from a name"), "Otto");
    await user.click(screen.getByRole("button", { name: "Seed from name" }));

    const expected = configForName("Otto");
    expect(screen.getByTestId("config").textContent).toBe(
      `${expected.shape}.${expected.color}.${expected.accessory}`,
    );
  });

  it("produces a valid config when surprised", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Surprise me" }));

    await waitFor(() => expect(onChange).toHaveBeenCalled());
    const config = onChange.mock.calls[0][0] as AvatarConfig;
    expect(isAccessoryId(config.accessory)).toBe(true);
    expect(FACETS[0].options.map((o) => o.id)).toContain(config.shape);
    expect(FACETS[1].options.map((o) => o.id)).toContain(config.color);
  });
});

describe("AvatarEditor helpers", () => {
  it("wraps around the option ring", () => {
    expect(nextIndex(0, 5, -1)).toBe(4);
    expect(nextIndex(4, 5, 1)).toBe(0);
  });

  it("falls back to the first option for an unknown value", () => {
    expect(
      selectedIndex(FACETS[0], {
        ...DEFAULT_CONFIG,
        shape: "nope" as AvatarConfig["shape"],
      }),
    ).toBe(0);
  });
});
