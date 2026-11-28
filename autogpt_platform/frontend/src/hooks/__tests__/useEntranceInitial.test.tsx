import { act, render, screen } from "@testing-library/react";
import { hydrateRoot, type Root } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, test } from "vitest";
import { useEntranceInitial } from "../useEntranceInitial";

function Probe() {
  const initial = useEntranceInitial("hidden");
  return <output>{String(initial)}</output>;
}

const roots: Root[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) act(() => root.unmount());
  document.body.innerHTML = "";
});

describe("useEntranceInitial", () => {
  test("the server's HTML and its hydration start at the end state", async () => {
    const container = document.createElement("div");
    container.innerHTML = renderToString(<Probe />);
    document.body.appendChild(container);
    expect(container.textContent).toBe("false");

    // A hydration render that disagreed with the server's HTML is reported here.
    const mismatches: unknown[] = [];
    await act(async () => {
      roots.push(
        hydrateRoot(container, <Probe />, {
          onRecoverableError: (error) => mismatches.push(error),
        }),
      );
    });

    expect(mismatches).toEqual([]);
    expect(container.textContent).toBe("hidden");
  });

  test("a component mounted after the page loaded gets its entrance", () => {
    render(<Probe />);
    expect(screen.getByRole("status").textContent).toBe("hidden");
  });
});
