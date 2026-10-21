import { describe, expect, test } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { AgentImageItem } from "../AgentImageItem";

function renderItem(image: string) {
  return render(
    <AgentImageItem
      image={image}
      index={0}
      playingVideoIndex={null}
      handlePlay={() => {}}
      handlePause={() => {}}
    />,
  );
}

describe("AgentImageItem", () => {
  test("renders the preview image while the URL loads", () => {
    renderItem("https://cdn.test/preview.png");
    expect(screen.getByAltText("Image")).toBeDefined();
  });

  test("replaces a dead preview URL with a placeholder", () => {
    const { container } = renderItem("https://cdn.test/dead.png");

    fireEvent.error(screen.getByAltText("Image"));

    expect(screen.queryByAltText("Image")).toBeNull();
    expect(container.querySelector("svg")).not.toBeNull();
  });
});
