import { describe, expect, test } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import Avatar, { AvatarFallback, AvatarImage } from "../Avatar";

function renderAvatar(src: string) {
  return render(
    <Avatar>
      <AvatarImage src={src} alt="Pwuts avatar" />
      <AvatarFallback>Pwuts</AvatarFallback>
    </Avatar>,
  );
}

describe("Avatar", () => {
  test("swaps a dead avatar URL for the generated fallback", () => {
    const { container } = renderAvatar("https://cdn.test/dead.png");

    expect(screen.getByAltText("Pwuts avatar")).toBeDefined();
    fireEvent.error(screen.getByAltText("Pwuts avatar"));

    expect(screen.queryByAltText("Pwuts avatar")).toBeNull();
    expect(container.querySelector("svg")).not.toBeNull();
  });

  test("shows the fallback when there is no avatar URL at all", () => {
    const { container } = render(
      <Avatar>
        <AvatarImage src="" alt="Pwuts avatar" />
        <AvatarFallback>Pwuts</AvatarFallback>
      </Avatar>,
    );

    expect(screen.queryByAltText("Pwuts avatar")).toBeNull();
    expect(container.querySelector("svg")).not.toBeNull();
  });
});
