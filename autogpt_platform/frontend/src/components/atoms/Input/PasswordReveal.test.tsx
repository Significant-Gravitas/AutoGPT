import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Input } from "./Input";

describe("password reveal", () => {
  it("toggles its input type, accessible name and pressed state on click", async () => {
    const user = userEvent.setup();
    render(
      <Input
        id="password"
        label="Password"
        type="password"
        defaultValue="secret"
      />,
    );
    const input = screen.getByLabelText("Password");
    const toggle = screen.getByRole("button", { name: "Show password" });
    expect(input.getAttribute("type")).toBe("password");
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    await user.click(toggle);
    expect(input.getAttribute("type")).toBe("text");
    expect(
      screen
        .getByRole("button", { name: "Hide password" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    await user.click(toggle);
    expect(input.getAttribute("type")).toBe("password");
    expect(toggle.getAttribute("aria-label")).toBe("Show password");
  });

  it.each([" ", "{Enter}"])(
    "supports keyboard activation with %j without submitting",
    async (key) => {
      const user = userEvent.setup();
      const submit = vi.fn((event) => event.preventDefault());
      render(
        <form onSubmit={submit}>
          <Input id="password" label="Password" type="password" />
          <button type="submit">Submit</button>
        </form>,
      );
      await user.tab();
      await user.tab();
      const toggle = screen.getByRole("button", { name: "Show password" });
      expect(document.activeElement).toBe(toggle);
      await user.keyboard(key);
      expect(screen.getByLabelText("Password").getAttribute("type")).toBe(
        "text",
      );
      expect(toggle.getAttribute("aria-pressed")).toBe("true");
      expect(submit).not.toHaveBeenCalled();
    },
  );

  it("stays revealed within the input group and masks when focus leaves", async () => {
    const user = userEvent.setup();
    const onBlur = vi.fn();
    render(
      <>
        <Input id="password" label="Password" type="password" onBlur={onBlur} />
        <button>Next</button>
      </>,
    );
    const input = screen.getByLabelText("Password");
    await user.click(screen.getByRole("button", { name: "Show password" }));
    await user.click(input);
    expect(input.getAttribute("type")).toBe("text");
    await user.tab();
    expect(input.getAttribute("type")).toBe("text");
    expect(onBlur).toHaveBeenCalledOnce();
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Next" }),
    );
    expect(input.getAttribute("type")).toBe("password");
  });

  it("masks when focus has no related target", async () => {
    const user = userEvent.setup();
    render(<Input id="password" label="Password" type="password" />);
    const toggle = screen.getByRole("button", { name: "Show password" });
    await user.click(toggle);
    act(() => {
      fireEvent.blur(toggle, { relatedTarget: null });
    });
    expect(screen.getByLabelText("Password").getAttribute("type")).toBe(
      "password",
    );
  });

  it("does not reveal a disabled password", async () => {
    const user = userEvent.setup();
    render(<Input id="password" label="Password" type="password" disabled />);
    const toggle = screen.getByRole("button", { name: "Show password" });
    expect(toggle.hasAttribute("disabled")).toBe(true);
    await user.click(toggle);
    expect(screen.getByLabelText("Password").getAttribute("type")).toBe(
      "password",
    );
  });

  it.each(["text", "email", "textarea", "amount", "number"] as const)(
    "does not add a reveal button for %s",
    (type) => {
      render(<Input id="field" label="Field" type={type} />);
      expect(screen.queryByRole("button", { name: /password/i })).toBeNull();
    },
  );
});

it("associates each password label and toggle with its own input ID", () => {
  render(
    <>
      <Input id="first-password" label="First password" type="password" />
      <Input id="second-password" label="Second password" type="password" />
    </>,
  );
  const first = screen.getByLabelText("First password");
  const second = screen.getByLabelText("Second password");
  expect(first.id).toBeTruthy();
  expect(first.id).not.toBe(second.id);
  expect(
    screen
      .getAllByRole("button", { name: "Show password" })
      .map((button) => button.getAttribute("aria-controls")),
  ).toEqual([first.id, second.id]);
});
