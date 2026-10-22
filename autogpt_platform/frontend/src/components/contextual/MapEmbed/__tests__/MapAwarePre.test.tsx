import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MapAwarePre } from "../MapAwarePre";

const SOURCE = JSON.stringify({
  title: "Cafes",
  pins: [{ name: "Cafe X", lat: 37.78, lng: -122.4 }],
});

describe("MapAwarePre", () => {
  afterEach(cleanup);

  it("shows pin links when no Maps key is configured", () => {
    render(
      <MapAwarePre>
        <code className="language-map">{SOURCE}</code>
      </MapAwarePre>,
    );
    expect(screen.getByTestId("map-embed")).toBeDefined();
    const link = screen.getByRole("link", { name: "Cafe X" });
    expect(link.getAttribute("href")).toContain("query=37.78%2C-122.4");
  });

  it("leaves other code blocks and broken map JSON untouched", () => {
    render(
      <MapAwarePre>
        <code className="language-map">{"{ nope"}</code>
      </MapAwarePre>,
    );
    expect(screen.queryByTestId("map-embed")).toBeNull();
    expect(screen.getByText("{ nope")).toBeDefined();
  });
});
