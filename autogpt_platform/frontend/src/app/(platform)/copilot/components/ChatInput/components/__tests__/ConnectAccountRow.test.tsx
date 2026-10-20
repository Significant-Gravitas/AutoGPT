import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { ConnectAccountRow } from "../ConnectionPicker/ConnectAccountRow";

it("keeps the entire connection row actionable and prevents duplicate connections while pending", async () => {
  const onConnect = vi.fn();
  const view = render(
    <ConnectAccountRow onConnect={onConnect} isConnecting={false} />,
  );
  await userEvent.click(screen.getByText("ChatGPT subscription"));
  expect(onConnect).toHaveBeenCalledTimes(1);

  view.rerender(<ConnectAccountRow onConnect={onConnect} isConnecting />);
  const connect = screen.getByRole("button", {
    name: "Connect a ChatGPT subscription",
  });
  expect(connect.hasAttribute("disabled")).toBe(true);
  const status = screen.getByRole("status", { name: "Connecting ChatGPT" });
  expect(status.querySelector("svg")?.classList.contains("animate-spin")).toBe(
    true,
  );
  await userEvent.click(connect);
  expect(onConnect).toHaveBeenCalledTimes(1);

  view.rerender(
    <ConnectAccountRow onConnect={onConnect} isConnecting={false} />,
  );
  expect(connect.hasAttribute("disabled")).toBe(false);
  expect(
    screen.queryByRole("status", { name: "Connecting ChatGPT" }),
  ).toBeNull();
});
