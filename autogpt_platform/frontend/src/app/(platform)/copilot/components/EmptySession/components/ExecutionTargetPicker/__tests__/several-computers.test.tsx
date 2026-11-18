import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import {
  getGetExperimentalListLocalExecutorsMockHandler200,
  getPostExperimentalListLocalExecutorDirectoriesMockHandler200,
} from "@/app/api/__generated__/endpoints/copilot/copilot.msw";
import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExecutionTargetPicker } from "../ExecutionTargetPicker";

const EXECUTORS_URL = "http://localhost:3000/api/proxy/api/copilot/executors";
// The list is asked for again every 3 s, so a wait for its next answer can
// take that long by itself.
const NEXT_CHECK = { timeout: 10_000 };

const WORKSTATION: ExecutorMachine = {
  machine_id: "1c49dced-6256-4f28-03af-f8f1ecf3eded",
  connection_id: "connection-1",
  display_name: "Workstation",
  platform: "windows",
  arch: "x86_64",
  shim_version: "0.2.0",
  capabilities: ["files"],
};
const LAPTOP: ExecutorMachine = {
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

function answerTheComputerList() {
  server.use(
    getGetExperimentalListLocalExecutorsMockHandler200(() => ({
      executors: connected,
    })),
  );
}

function failTheComputerList() {
  server.use(
    http.get(EXECUTORS_URL, () =>
      HttpResponse.json({ detail: "unavailable" }, { status: 500 }),
    ),
  );
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

async function openTheSetupSteps(user: User) {
  await user.click(
    screen.getByRole("button", { name: "Connect Another Computer" }),
  );
  await screen.findByText("Connect a Computer");
}

// The list's own 3 s check may get there first and take the button away.
async function clickIfStillThere(user: User, name: string) {
  const button = screen.queryByRole("button", { name });
  if (button) await user.click(button);
}

async function untilACheckHasFailed(user: User) {
  failTheComputerList();
  await clickIfStillThere(user, "Check Again");
  await screen.findByText(
    "Could Not Check Your Computers",
    undefined,
    NEXT_CHECK,
  );
}

function chosenComputer() {
  return screen.getByRole("combobox").textContent;
}

describe("the folder dialog with more than one computer", () => {
  beforeEach(() => {
    resetTargetStore();
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
    connected = [WORKSTATION];
    answerTheComputerList();
    server.use(
      getPostExperimentalListLocalExecutorDirectoriesMockHandler200(
        ({ params }) => ({
          connection_id:
            params.machineId === LAPTOP.machine_id
              ? LAPTOP.connection_id
              : WORKSTATION.connection_id,
          browse_id: "browse-1",
          current: null,
          parent_ref: null,
          entries: [],
          next_cursor: null,
          truncated: false,
          expires_at: 1,
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
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await clickIfStillThere(user, "Check Again");

    expect(
      await screen.findByText(
        "Laptop connected. Choose it under Computer.",
        undefined,
        NEXT_CHECK,
      ),
    ).toBeDefined();
    expect(screen.queryByText("Connect a Computer")).toBeNull();
    expect(chosenComputer()).toBe("Workstation · Windows");
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toMatchObject({
      kind: "local",
      machineID: WORKSTATION.machine_id,
      connectionID: WORKSTATION.connection_id,
    });
  });

  test("takes the notice away when that computer disconnects again", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await clickIfStillThere(user, "Check Again");
    await screen.findByText(/^Laptop connected\./, undefined, NEXT_CHECK);

    connected = [WORKSTATION];
    await openTheSetupSteps(user);
    await clickIfStillThere(user, "Check Again");

    await waitFor(
      () => expect(screen.queryByText(/^Laptop connected\./)).toBeNull(),
      NEXT_CHECK,
    );
  });

  test("keeps the steps open through a check that failed", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    await untilACheckHasFailed(user);
    answerTheComputerList();
    await clickIfStillThere(user, "Retry");

    expect(
      await screen.findByText("Connect a Computer", undefined, NEXT_CHECK),
    ).toBeDefined();
  });

  test("says so when another computer connected while a check was failing", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    await untilACheckHasFailed(user);
    connected = [WORKSTATION, LAPTOP];
    answerTheComputerList();
    await clickIfStillThere(user, "Retry");

    expect(
      await screen.findByText(
        "Laptop connected. Choose it under Computer.",
        undefined,
        NEXT_CHECK,
      ),
    ).toBeDefined();
    expect(screen.queryByText("Connect a Computer")).toBeNull();
  });

  test("puts the steps away when the dialog closes", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await user.click(
      screen.getByRole("button", { name: "Choose a Local PC folder" }),
    );
    await screen.findByText("This folder has no subfolders.");

    expect(screen.queryByText("Connect a Computer")).toBeNull();
    expect(
      screen
        .getByRole("button", { name: "Connect Another Computer" })
        .getAttribute("aria-expanded"),
    ).toBe("false");
  });

  test("tells apart two computers that share a name", async () => {
    const user = userEvent.setup();
    connected = [
      WORKSTATION,
      {
        ...WORKSTATION,
        machine_id: LAPTOP.machine_id,
        connection_id: LAPTOP.connection_id,
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
