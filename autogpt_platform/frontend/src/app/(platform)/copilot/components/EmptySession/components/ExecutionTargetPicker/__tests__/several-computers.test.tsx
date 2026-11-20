import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import {
  getGetExperimentalListLocalExecutorsQueryKey,
  type getExperimentalListLocalExecutorsResponse,
} from "@/app/api/__generated__/endpoints/copilot/copilot";
import {
  getGetExperimentalListLocalExecutorsMockHandler200,
  getPostExperimentalListLocalExecutorDirectoriesMockHandler200,
} from "@/app/api/__generated__/endpoints/copilot/copilot.msw";
import type { DirectoryListRequest } from "@/app/api/__generated__/models/directoryListRequest";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExecutionTargetPicker } from "../ExecutionTargetPicker";
import { LAPTOP, STUDIO, WORKSTATION } from "./computers";

const EXECUTORS_URL = "http://localhost:3000/api/proxy/api/copilot/executors";
const CHECK_INTERVAL_MS = 3_000;
const LAPTOP_CONNECTED = "Laptop · macOS connected. Choose it under Computer.";

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

async function checkAgain(user: User) {
  await user.click(screen.getByRole("button", { name: "Check Again" }));
}

async function untilACheckHasFailed(user: User) {
  failTheComputerList();
  await checkAgain(user);
  await screen.findByText("Could Not Check Your Computers");
}

async function retry(user: User) {
  await user.click(screen.getByRole("button", { name: "Retry" }));
}

function chosenComputer() {
  return screen.getByRole("combobox").textContent;
}

async function choose(user: User, name: string) {
  await user.click(screen.getByRole("combobox"));
  await user.click(await screen.findByRole("option", { name }));
  await waitFor(() => expect(chosenComputer()).toBe(name));
  await screen.findByText("This folder has no subfolders.");
}

// The region screen readers hear from. It is in the dialog whatever the
// dialog shows, since one that arrives already filled isn't read out.
function readOutRegion() {
  const region = document.querySelector('.sr-only[aria-live="polite"]');
  if (!region) throw new Error("The dialog has no region to read out from");
  return region;
}

// Once on screen beside the control, once in the region that reads it out.
async function seesTheNotice(text: string) {
  expect(await screen.findAllByText(text)).toHaveLength(2);
  expect(readOutRegion().textContent).toBe(text);
}

describe("the folder dialog with more than one computer", () => {
  beforeEach(() => {
    // The dialog asks for the list again every 3 s. Here that check runs only
    // when a test runs it, so no test depends on when it would have fired.
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    resetTargetStore();
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
    connected = [WORKSTATION];
    answerTheComputerList();
    server.use(
      getPostExperimentalListLocalExecutorDirectoriesMockHandler200(
        async ({ request }) => ({
          connection_id: ((await request.json()) as DirectoryListRequest)
            .expected_connection_id,
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
    vi.useRealTimers();
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
    expect(document.activeElement).toBe(connectAnother);
  });

  test("says when another computer connects, and leaves the chosen one alone", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await checkAgain(user);

    await seesTheNotice(LAPTOP_CONNECTED);
    expect(screen.queryByText("Connect a Computer")).toBeNull();
    expect(chosenComputer()).toBe("Workstation · Windows");
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toMatchObject({
      kind: "local",
      machineID: WORKSTATION.machine_id,
      connectionID: WORKSTATION.connection_id,
    });
  });

  test("moves focus to the Computer menu when the steps close under it", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await checkAgain(user);
    await seesTheNotice(LAPTOP_CONNECTED);

    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole("combobox")),
    );
  });

  test("leaves focus alone when a computer connects while it is elsewhere", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    // Focus goes into the steps and out again before the computer connects.
    screen.getByRole("link", { name: /^Install pipx/ }).focus();
    const cancel = screen.getByRole("button", { name: "Cancel" });
    cancel.focus();
    connected = [WORKSTATION, LAPTOP];
    await act(() => vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS));
    await seesTheNotice(LAPTOP_CONNECTED);

    expect(screen.queryByText("Connect a Computer")).toBeNull();
    expect(document.activeElement).toBe(cancel);
  });

  test("names a computer that connects the way the menu does", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [
      WORKSTATION,
      {
        ...WORKSTATION,
        machine_id: LAPTOP.machine_id,
        connection_id: LAPTOP.connection_id,
      },
    ];
    await checkAgain(user);

    await seesTheNotice(
      "Workstation · Windows · ID 36d78179 connected. Choose it under Computer.",
    );
    await user.click(screen.getByRole("combobox"));
    expect(
      (await screen.findAllByRole("option")).map(
        (option) => option.textContent,
      ),
    ).toEqual([
      "Workstation · Windows · ID 1c49dced",
      "Workstation · Windows · ID 36d78179",
    ]);
  });

  test("names each computer when two connect at once", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP, STUDIO];
    await checkAgain(user);

    await seesTheNotice(
      "Laptop · macOS and Studio · macOS connected. Choose one under Computer.",
    );
  });

  test("keeps naming a computer that joined earlier when another joins", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await checkAgain(user);
    await seesTheNotice(LAPTOP_CONNECTED);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP, STUDIO];
    await checkAgain(user);

    await seesTheNotice(
      "Laptop · macOS and Studio · macOS connected. Choose one under Computer.",
    );
  });

  test("stops naming a computer once it has been chosen", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await checkAgain(user);
    await seesTheNotice(LAPTOP_CONNECTED);

    await choose(user, "Laptop · macOS");
    expect(screen.queryAllByText(LAPTOP_CONNECTED)).toHaveLength(0);
    await choose(user, "Workstation · Windows");

    expect(screen.queryAllByText(LAPTOP_CONNECTED)).toHaveLength(0);
  });

  test("names a computer that connects at the moment another is chosen", async () => {
    const user = userEvent.setup();
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={client}>
        <ExecutionTargetPicker />
      </QueryClientProvider>,
    );
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await checkAgain(user);
    await seesTheNotice(LAPTOP_CONNECTED);

    // The list's next answer and the person's choice reach one render. A
    // click and a network answer can't be made to land together, so both are
    // put where the picker reads them.
    connected = [WORKSTATION, LAPTOP, STUDIO];
    act(() => {
      client.setQueryData<getExperimentalListLocalExecutorsResponse>(
        getGetExperimentalListLocalExecutorsQueryKey(),
        (answer) =>
          answer?.status === 200
            ? { ...answer, data: { executors: connected } }
            : answer,
      );
      useCopilotUIStore.getState().setNewChatExecutionTarget({
        kind: "local",
        machineID: LAPTOP.machine_id,
        machineLabel: LAPTOP.display_name,
        connectionID: LAPTOP.connection_id,
        browseID: null,
        directoryRef: null,
        displayPath: null,
      });
    });

    await seesTheNotice("Studio · macOS connected. Choose it under Computer.");
    expect(chosenComputer()).toBe("Laptop · macOS");
  });

  test("takes the notice away only when that computer disconnects", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [WORKSTATION, LAPTOP];
    await checkAgain(user);
    await seesTheNotice(LAPTOP_CONNECTED);

    // Opening the steps again leaves it: only the list takes it away.
    await openTheSetupSteps(user);
    await seesTheNotice(LAPTOP_CONNECTED);
    connected = [WORKSTATION];
    await checkAgain(user);

    await waitFor(() =>
      expect(screen.queryAllByText(LAPTOP_CONNECTED)).toHaveLength(0),
    );
  });

  test("keeps the steps open through a check that failed", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    await untilACheckHasFailed(user);
    answerTheComputerList();
    await retry(user);

    expect(await screen.findByText("Connect a Computer")).toBeDefined();
  });

  test("says so when another computer connected while a check was failing", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    const regionBeforeTheFailure = readOutRegion();
    await untilACheckHasFailed(user);
    connected = [WORKSTATION, LAPTOP];
    answerTheComputerList();
    await retry(user);

    await seesTheNotice(LAPTOP_CONNECTED);
    expect(readOutRegion()).toBe(regionBeforeTheFailure);
    expect(screen.queryByText("Connect a Computer")).toBeNull();
  });

  test("still says so when the chosen computer reconnected in the same check", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [{ ...WORKSTATION, connection_id: "connection-1b" }, LAPTOP];
    await checkAgain(user);

    await seesTheNotice(LAPTOP_CONNECTED);
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toMatchObject({
      machineID: WORKSTATION.machine_id,
      connectionID: "connection-1b",
    });
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

  test("starts with the steps closed again once the last computer has gone", async () => {
    const user = userEvent.setup();

    render(<ExecutionTargetPicker />);
    await openTheFolderDialog(user);
    await openTheSetupSteps(user);
    connected = [];
    await checkAgain(user);
    await waitFor(() => expect(screen.queryByRole("combobox")).toBeNull());
    connected = [WORKSTATION];
    await checkAgain(user);
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
      expect(chosenComputer()).toBe("Workstation · Windows · ID 1c49dced"),
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
