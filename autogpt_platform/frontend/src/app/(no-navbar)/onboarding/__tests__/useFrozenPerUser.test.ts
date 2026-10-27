import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useFrozenPerUser } from "../useFrozenPerUser";

function render(initial: {
  userID: string | null;
  ready: boolean;
  value: string;
}) {
  return renderHook(
    ({ userID, ready, value }) => useFrozenPerUser(userID, ready, () => value),
    { initialProps: initial },
  );
}

describe("useFrozenPerUser", () => {
  it("answers nothing until ready", () => {
    const { result } = render({ userID: "user-1", ready: false, value: "a" });

    expect(result.current).toBeNull();
  });

  it("keeps the first answer when what it reads changes later", () => {
    const { result, rerender } = render({
      userID: "user-1",
      ready: true,
      value: "a",
    });

    rerender({ userID: "user-1", ready: true, value: "b" });

    expect(result.current).toBe("a");
  });

  it("answers afresh for another account", () => {
    const { result, rerender } = render({
      userID: "user-1",
      ready: true,
      value: "a",
    });

    rerender({ userID: "user-2", ready: false, value: "b" });
    expect(result.current).toBeNull();
    rerender({ userID: "user-2", ready: true, value: "b" });
    expect(result.current).toBe("b");
  });
});
