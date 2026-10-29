import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MessageResponse } from "../Message";

const MAP_REPLY = [
  "Here you go:",
  "",
  "```map",
  JSON.stringify({
    title: "Cafes",
    pins: [{ name: "Cafe X", lat: 37.78, lng: -122.4 }],
  }),
  "```",
].join("\n");

describe("MessageResponse map blocks", () => {
  afterEach(cleanup);

  it("renders a map block as a map instead of code", async () => {
    const { container } = render(
      <MessageResponse>{MAP_REPLY}</MessageResponse>,
    );
    await waitFor(() => expect(screen.getByTestId("map-embed")).toBeDefined());
    expect(screen.getByText("Cafes")).toBeDefined();
    expect(
      container.querySelector("[data-streamdown='code-block']"),
    ).toBeNull();
  });

  it("keeps other code blocks as code", async () => {
    const { container } = render(
      <MessageResponse>{"```js\nconst a = 1;\n```"}</MessageResponse>,
    );
    await waitFor(() =>
      expect(
        container.querySelector("[data-streamdown='code-block']"),
      ).not.toBeNull(),
    );
    expect(screen.queryByTestId("map-embed")).toBeNull();
  });
});
