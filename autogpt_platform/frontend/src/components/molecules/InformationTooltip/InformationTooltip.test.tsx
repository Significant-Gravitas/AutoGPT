import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { InformationTooltip } from "./InformationTooltip";

it("keeps Markdown links available when moving from the trigger to click them", async () => {
  const user = userEvent.setup();
  render(
    <InformationTooltip description="Visit our [documentation](https://docs.example.com) for more details." />,
  );

  await user.hover(screen.getByRole("button", { name: "More information" }));
  const [link] = await screen.findAllByRole("link", { name: "documentation" });
  await user.hover(link);

  expect(link.isConnected).toBe(true);
  expect(link.getAttribute("href")).toBe("https://docs.example.com");
  expect(link.getAttribute("target")).toBe("_blank");

  const onClick = vi.fn((event: Event) => event.preventDefault());
  link.addEventListener("click", onClick);
  await user.click(link);
  expect(onClick).toHaveBeenCalledOnce();
});
