import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "../ChatInput";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => ({
  ...(await importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));

afterEach(() => {
  vi.clearAllMocks();
});

// Assembled at run time so no secret scanner mistakes it for a leaked token.
const GITHUB_TOKEN = "ghp" + "_" + "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1eF3hJ5";
const WARNING = /this looks like an api key or other secret/i;

function composer() {
  return screen.getByRole("textbox") as HTMLTextAreaElement;
}

describe("ChatInput secret warning", () => {
  it("warns about a pasted GitHub token and keeps the text as pasted", async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} />);

    await user.click(composer());
    await user.paste(`use this token ${GITHUB_TOKEN}`);

    const warning = await screen.findByText(WARNING);
    const link = screen.getByRole("link", { name: "Integrations" });
    expect(link.getAttribute("href")).toBe("/settings/integrations");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(warning.textContent).toContain("Otto");
    expect(composer().value).toBe(`use this token ${GITHUB_TOKEN}`);
  });

  it("puts the warning into a status region that was already there, so it is announced", async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} />);

    const region = screen.getByTestId("secret-paste-notice");
    expect(region.getAttribute("role")).toBe("status");
    expect(region.textContent).toBe("");

    await user.click(composer());
    await user.paste(GITHUB_TOKEN);

    const warning = await screen.findByText(WARNING);
    expect(screen.getByTestId("secret-paste-notice")).toBe(region);
    expect(region.contains(warning)).toBe(true);
  });

  it("names the expert the chat is with instead of Otto", async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} assistantName="Maria" />);

    await user.click(composer());
    await user.paste(GITHUB_TOKEN);

    const warning = await screen.findByText(WARNING);
    expect(warning.textContent).toContain("To let Maria use a key");
    expect(warning.textContent).not.toContain("Otto");
  });

  it("drops the warning once the token is removed", async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} />);

    await user.click(composer());
    await user.paste(GITHUB_TOKEN);
    await screen.findByText(WARNING);

    await user.clear(composer());
    await user.type(composer(), "never mind");

    await waitFor(() => expect(screen.queryByText(WARNING)).toBeNull());
  });

  it("stays quiet for ordinary text", async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} />);

    await user.type(
      composer(),
      "check commit 3f786850e387550fdab836ed7e6dc881de23001b",
    );

    expect(composer().value).toContain("3f786850");
    expect(screen.queryByText(WARNING)).toBeNull();
  });

  it("still sends a message that holds a token", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);

    await user.click(composer());
    await user.paste(`token ${GITHUB_TOKEN}`);
    await screen.findByText(WARNING);
    await user.keyboard("{Enter}");

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toBe(`token ${GITHUB_TOKEN}`);
    await waitFor(() => expect(screen.queryByText(WARNING)).toBeNull());
  });
});
