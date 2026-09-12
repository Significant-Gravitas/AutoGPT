import { describe, expect, test } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { AgentImages } from "../AgentImage";

const IMAGES = ["https://cdn.test/one.png", "https://cdn.test/two.png"];

describe("AgentImages thumbnails", () => {
  test("renders a thumbnail per image", () => {
    render(<AgentImages images={IMAGES} />);
    expect(screen.getByAltText("Thumbnail 1")).toBeDefined();
    expect(screen.getByAltText("Thumbnail 2")).toBeDefined();
  });

  test("replaces only the dead thumbnail with a placeholder", () => {
    render(<AgentImages images={IMAGES} />);

    fireEvent.error(screen.getByAltText("Thumbnail 2"));

    expect(screen.queryByAltText("Thumbnail 2")).toBeNull();
    expect(screen.getByAltText("Thumbnail 1")).toBeDefined();
  });
});
