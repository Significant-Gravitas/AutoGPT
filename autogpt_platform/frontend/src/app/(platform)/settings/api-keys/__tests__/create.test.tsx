import { describe, expect, test } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV1ListUserApiKeysMockHandler,
  getPostV1CreateNewApiKeyMockHandler200,
  getPostV1CreateNewApiKeyMockHandler422,
  getPostV1CreateNewApiKeyResponseMock200,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";

import SettingsApiKeysPage from "../page";

function openCreateDialog() {
  const createButtons = screen.getAllByRole("button", {
    name: /^create key$/i,
  });
  fireEvent.click(createButtons[0]);
}

describe("SettingsApiKeysPage - create flow", () => {
  test("opens the create dialog with a form when Create Key is clicked", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/create api key/i)).toBeDefined();
    expect(within(dialog).getByLabelText(/^name$/i)).toBeDefined();
    expect(within(dialog).getByLabelText(/description/i)).toBeDefined();
    expect(within(dialog).getByText(/permissions/i)).toBeDefined();
  });

  test("disables submit until the form is valid", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    const submit = within(dialog).getByRole("button", { name: /create key/i });
    expect((submit as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "Integration Test Key" },
    });
    // Still invalid — no permission picked yet.
    await waitFor(() => {
      expect((submit as HTMLButtonElement).disabled).toBe(true);
    });

    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    await waitFor(() => {
      expect((submit as HTMLButtonElement).disabled).toBe(false);
    });
  });

  test("submits successfully and switches to the success view with plain text key", async () => {
    const plain = "plain-secret-key-abc123";
    server.use(
      getGetV1ListUserApiKeysMockHandler([]),
      getPostV1CreateNewApiKeyMockHandler200(
        getPostV1CreateNewApiKeyResponseMock200({ plain_text_key: plain }),
      ),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "My New Key" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    fireEvent.click(
      within(dialog).getByRole("button", { name: /create key/i }),
    );

    expect(await screen.findByText(/your new api key/i)).toBeDefined();
    expect(await screen.findByText(plain)).toBeDefined();
    expect(screen.getByRole("button", { name: /^close$/i })).toBeDefined();
  });

  test("keeps the form open when the API returns 422", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([]),
      getPostV1CreateNewApiKeyMockHandler422(),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "Should Fail" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    fireEvent.click(
      within(dialog).getByRole("button", { name: /create key/i }),
    );

    // No transition to the success view — form inputs remain mounted.
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/^name$/i)).toBeDefined();
    });
    expect(screen.queryByText(/your new api key/i)).toBeNull();
  });

  test("reopening the dialog after a successful create resets the form", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([]),
      getPostV1CreateNewApiKeyMockHandler200(
        getPostV1CreateNewApiKeyResponseMock200({
          plain_text_key: "reset-me",
        }),
      ),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    let dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "First Key" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: /create key/i }),
    );

    await screen.findByText("reset-me");
    fireEvent.click(screen.getByRole("button", { name: /^close$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    openCreateDialog();
    dialog = await screen.findByRole("dialog");

    const nameInput = within(dialog).getByLabelText(
      /^name$/i,
    ) as HTMLInputElement;
    expect(nameInput.value).toBe("");
    expect(within(dialog).queryByText(/your new api key/i)).toBeNull();
  });
});
