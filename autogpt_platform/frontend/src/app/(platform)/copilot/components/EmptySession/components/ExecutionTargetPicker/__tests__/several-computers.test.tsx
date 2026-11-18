import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExecutionTargetPicker } from "../ExecutionTargetPicker";

const EXECUTORS_URL = "http://localhost:3000/api/proxy/api/copilot/executors";
const DIRECTORIES_URL = `${EXECUTORS_URL}/:machineID/directories`;

const WORKSTATION = {
  machine_id: "1c49dced-6256-4f28-03af-f8f1ecf3eded",
  connection_id: "connection-1",
  display_name: "Workstation",
  platform: "windows",
  arch: "x86_64",
  shim_version: "0.2.0",
  capabilities: ["files"],
};
const LAPTOP = {
  ...WORKSTATION,
  machine_id: "36d78179-05b7-7f11-32dd-34f3999eceeb",
  connection_id: "connection-2",
  display_name: "Laptop",
  platform: "darwin",
  arch: "arm64",
};

type User = ReturnType<typeof userEvent.setup>;

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-1" } }),
}));

let connected = [WORKSTATION];

function resetTargetStore() {
  useCopilotUIStore.setState({
    newChatExecutionTarget: { kind: "cloud" },
    isExecutionTargetPickerOpen: false,
    executionTargetError: null,
  });
}

async function openTheFolderDialog(user: User) {
  await user.click(
    screen.getByRole("button", { name: "Execution target: Cloud" }),
  );
  await user.click(
    await screen.findByRole("menuitemradio", { name: /^Local PC/i }),
  );
  await screen.findByRole("dialog", {
    name: /^Choose a Folder on Your Local PC$/i,
  });
  // The folder list of the chosen computer has loaded.
  await screen.findByText("This folder has no subfolders.");
}

function chosenComputer() {
  return screen.getByRole("combobox").textContent;
}

describe("the folder dialog with more than one computer", () => {
  beforeEach(() => {
    resetTargetStore();
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
    connected = [WORKSTATION];
    server.use(
      http.get(EXECUTORS_URL, () =>
        HttpResponse.json({ executors: connected }),
      ),
      http.post(DIRECTORIES_URL, ({ params }) =>
        HttpResponse.json({
          connection_id:
            params.machineID === LAPTOP.machine_id
              ? "connection-2"
              : "connection-1",
          browse_id: "browse-1",
          current: null,
          parent_ref: null,
          entries: [],
          truncated: false,
          expires_at: "2026-07-10T18:00:00Z",
        }),
      ),
    );
  });

  afterEach(() => {
    storage.clean(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
    resetTargetStore();
  });

  test("the setup steps can be brought up while a computer is connected", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);

    expect(chosenComputer()).toBe("Workstation · Windows");
    expect(screen.queryByText("Connect a Computer")).toBeNull();
    const connectAnother = screen.getByRole("button", {
      name: "Connect Another Computer",
    });
    expect(connectAnother.getAttribute("aria-expanded")).toBe("false");

    await user.click(connectAnother);

    expect(await screen.findByText("Connect a Computer")).toBeDefined();
    expect(screen.getByText(/^pipx install git\+https:/)).toBeDefined();
    expect(connectAnother.getAttribute("aria-expanded")).toBe("true");
    // The chosen computer's folders stay in reach beside the steps.
    expect(screen.getByText("This folder has no subfolders.")).toBeDefined();

    await user.click(connectAnother);

    expect(screen.queryByText("Connect a Computer")).toBeNull();
  });

  test("says when another computer connects, and leaves the chosen one alone", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await user.click(
      screen.getByRole("button", { name: "Connect Another Computer" }),
    );
    connected = [WORKSTATION, LAPTOP];
    await user.click(
      await screen.findByRole("button", { name: "Check Again" }),
    );

    expect(
      await screen.findByText("Laptop connected. Choose it under Computer."),
    ).toBeDefined();
    expect(screen.queryByText("Connect a Computer")).toBeNull();
    expect(chosenComputer()).toBe("Workstation · Windows");
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toMatchObject({
      kind: "local",
      machineID: WORKSTATION.machine_id,
      connectionID: "connection-1",
    });
  });

  test("tells apart two computers that share a name", async () => {
    const user = userEvent.setup();
    connected = [
      WORKSTATION,
      {
        ...WORKSTATION,
        machine_id: LAPTOP.machine_id,
        connection_id: "connection-2",
      },
    ];

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);

    await waitFor(() =>
      expect(chosenComputer()).toBe("Workstation · Windows · 1c49dced"),
    );
    expect(
      screen.getByText(/^Some of your computers have the same name\./),
    ).toBeDefined();
  });

  test("leaves the names as they are when they differ", async () => {
    const user = userEvent.setup();
    connected = [WORKSTATION, LAPTOP];

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);

    expect(chosenComputer()).toBe("Workstation · Windows");
    expect(screen.queryByText(/have the same name/)).toBeNull();
  });
});
