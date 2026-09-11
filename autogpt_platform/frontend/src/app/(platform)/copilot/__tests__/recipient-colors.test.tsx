import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RecipientChip } from "../components/ChatInput/components/RecipientChip";
import { useRecipientPicker } from "../components/EmptySession/useRecipientPicker";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => true,
}));
vi.mock("@/components/molecules/BotAvatar/BotAvatar", () => ({
  BotAvatar: ({
    config,
    title,
  }: {
    config: { color: string };
    title: string;
  }) => <span role="img" aria-label={title} data-color={config.color} />,
}));

function Picker() {
  const { recipient, options, selectRecipient } = useRecipientPicker();
  return (
    <RecipientChip
      recipient={recipient}
      options={options}
      onSelect={selectRecipient}
    />
  );
}

describe("recipient colors", () => {
  it("keeps roster colors in the selected avatar and menu", async () => {
    server.use(
      http.get("*/api/experts/identities", () =>
        HttpResponse.json([
          {
            id: "expert-maria",
            name: "Maria",
            color: "orange-500",
            avatar_url: null,
            role: "Marketing",
            is_archived: false,
          },
        ]),
      ),
    );
    const Wrapper = withNuqsTestingAdapter({
      searchParams: "?expertId=expert-maria",
      hasMemory: true,
    });
    render(
      <Wrapper>
        <Picker />
      </Wrapper>,
    );
    const chip = await screen.findByRole("button", {
      name: /Sending to Maria/,
    });
    await waitFor(() =>
      expect(within(chip).getByRole("img").getAttribute("data-color")).toBe(
        "coral",
      ),
    );
    await userEvent.click(chip);
    const menu = await screen.findByRole("menu");
    expect(
      within(menu)
        .getByRole("img", { name: "Maria" })
        .getAttribute("data-color"),
    ).toBe("coral");
  });
});
