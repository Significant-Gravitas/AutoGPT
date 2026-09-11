import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ExecutionTargetPicker } from "../ExecutionTargetPicker";

const EXECUTORS_URL = "http://localhost:3000/api/proxy/api/copilot/executors";
const DIRECTORIES_URL =
  "http://localhost:3000/api/proxy/api/copilot/executors/machine-1/directories";

const MACHINE = {
  machine_id: "machine-1",
  connection_id: "connection-1",
  display_name: "Workstation",
  platform: "windows",
  arch: "x86_64",
  shim_version: "0.2.0",
  capabilities: ["files", "shell"],
};

function resetTargetStore() {
  useCopilotUIStore.setState({
    newChatExecutionTarget: { kind: "cloud" },
    isExecutionTargetPickerOpen: false,
    executionTargetError: null,
  });
}

function mockMachines(executors = [MACHINE]) {
  server.use(http.get(EXECUTORS_URL, () => HttpResponse.json({ executors })));
}

describe("ExecutionTargetPicker", () => {
  beforeEach(() => {
    resetTargetStore();
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "true");
  });

  afterEach(() => {
    storage.clean(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
    resetTargetStore();
    vi.restoreAllMocks();
  });

  it("defaults to Cloud and selects Local PC with the keyboard", async () => {
    const user = userEvent.setup();
    mockMachines();
    server.use(
      http.post(DIRECTORIES_URL, () =>
        HttpResponse.json({
          connection_id: "connection-1",
          browse_id: "browse-1",
          current: null,
          parent_ref: null,
          entries: [],
          truncated: false,
          expires_at: "2026-07-10T18:00:00Z",
        }),
      ),
    );

    render(<ExecutionTargetPicker />);

    const trigger = screen.getByRole("button", {
      name: "Execution target: Cloud",
    });
    trigger.focus();
    await user.keyboard("{Enter}");
    const localOption = await screen.findByRole("menuitemradio", {
      name: /Local PC/i,
    });
    localOption.focus();
    await user.keyboard("{Enter}");

    expect(
      await screen.findByRole("dialog", {
        name: /Choose a Folder on Your Local PC/i,
      }),
    ).toBeDefined();
    expect(useCopilotUIStore.getState().newChatExecutionTarget.kind).toBe(
      "local",
    );
  });

  it("browses remote folders and requires explicit folder confirmation", async () => {
    const user = userEvent.setup();
    mockMachines();
    const requests: Array<Record<string, unknown>> = [];
    server.use(
      http.post(DIRECTORIES_URL, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        requests.push(body);
        if (body.directory_ref === "projects-ref") {
          return HttpResponse.json({
            connection_id: "connection-1",
            browse_id: "browse-1",
            current: {
              directory_ref: "projects-ref",
              name: "Projects",
              path: "C:\\Users\\Ada\\Projects",
            },
            parent_ref: null,
            entries: [
              {
                directory_ref: "autogpt-ref",
                name: "AutoGPT",
                path: "C:\\Users\\Ada\\Projects\\AutoGPT",
              },
            ],
            truncated: false,
            expires_at: "2026-07-10T18:00:00Z",
          });
        }
        return HttpResponse.json({
          connection_id: "connection-1",
          browse_id: "browse-1",
          current: null,
          parent_ref: null,
          entries: [
            {
              directory_ref: "projects-ref",
              name: "Projects",
              path: "C:\\Users\\Ada\\Projects",
            },
          ],
          truncated: false,
          expires_at: "2026-07-10T18:00:00Z",
        });
      }),
    );

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);
    await user.click(await screen.findByRole("button", { name: "Projects" }));

    expect(await screen.findByText("C:\\Users\\Ada\\Projects")).toBeDefined();
    expect(
      (
        screen.getByRole("button", {
          name: "Use This Folder",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
    await user.click(screen.getByRole("button", { name: "Use This Folder" }));

    expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual({
      kind: "local",
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "connection-1",
      browseID: "browse-1",
      directoryRef: "projects-ref",
      displayPath: "C:\\Users\\Ada\\Projects",
    });
    expect(requests).toEqual([
      {
        expected_connection_id: "connection-1",
        browse_id: null,
        directory_ref: null,
      },
      {
        expected_connection_id: "connection-1",
        browse_id: "browse-1",
        directory_ref: "projects-ref",
      },
    ]);
  });

  it("loads another host-owned page of folders", async () => {
    const user = userEvent.setup();
    mockMachines();
    const requests: Array<Record<string, unknown>> = [];
    server.use(
      http.post(DIRECTORIES_URL, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        requests.push(body);
        if (body.cursor === "next-page") {
          return HttpResponse.json({
            connection_id: "connection-1",
            browse_id: "browse-1",
            current: {
              directory_ref: "home-ref",
              name: "Home",
              path: "C:\\Users\\Ada",
            },
            parent_ref: null,
            entries: [
              {
                directory_ref: "second-ref",
                name: "Second Page",
                path: "C:\\Users\\Ada\\Second Page",
              },
            ],
            next_cursor: null,
            truncated: false,
            expires_at: 1,
          });
        }
        if (body.directory_ref === "home-ref") {
          return HttpResponse.json({
            connection_id: "connection-1",
            browse_id: "browse-1",
            current: {
              directory_ref: "home-ref",
              name: "Home",
              path: "C:\\Users\\Ada",
            },
            parent_ref: null,
            entries: [
              {
                directory_ref: "first-ref",
                name: "First Page",
                path: "C:\\Users\\Ada\\First Page",
              },
            ],
            next_cursor: "next-page",
            truncated: false,
            expires_at: 1,
          });
        }
        return HttpResponse.json({
          connection_id: "connection-1",
          browse_id: "browse-1",
          current: null,
          parent_ref: null,
          entries: [
            {
              directory_ref: "home-ref",
              name: "Home",
              path: "C:\\Users\\Ada",
            },
          ],
          next_cursor: null,
          truncated: false,
          expires_at: 1,
        });
      }),
    );

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);
    await user.click(await screen.findByRole("button", { name: "Home" }));
    await user.click(
      await screen.findByRole("button", { name: "Load More Folders" }),
    );

    expect(
      await screen.findByRole("button", { name: "First Page" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Second Page" })).toBeDefined();
    expect(requests.at(-1)).toMatchObject({
      browse_id: "browse-1",
      directory_ref: "home-ref",
      cursor: "next-page",
    });
  });

  it("teaches setup and keeps polling when no computer is connected", async () => {
    const user = userEvent.setup();
    mockMachines([]);

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);

    expect(await screen.findByText("Connect a Computer")).toBeDefined();
    expect(
      screen.getByText(
        "pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git",
      ),
    ).toBeDefined();
    expect(screen.getByText("autogpt-shim auth")).toBeDefined();
    expect(screen.getByText("autogpt-shim install")).toBeDefined();
    expect(
      screen.getByText(/even when this chat is open on another device/i),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Check Again" })).toBeDefined();
  });

  it("surfaces stale directory failures and offers retry", async () => {
    const user = userEvent.setup();
    mockMachines();
    const requests: unknown[] = [];
    server.use(
      http.post(DIRECTORIES_URL, async ({ request }) => {
        requests.push(await request.json());
        return HttpResponse.json(
          { detail: "Executor connection changed" },
          { status: 409 },
        );
      }),
    );

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);

    expect(
      await screen.findAllByText(/computer or folder changed/i),
    ).not.toHaveLength(0);
    expect(screen.getByRole("button", { name: "Retry" })).toBeDefined();
    expect(useCopilotUIStore.getState().executionTargetError).toMatch(
      /changed/i,
    );
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests).toEqual([
      {
        expected_connection_id: "connection-1",
        browse_id: null,
        directory_ref: null,
      },
      {
        expected_connection_id: "connection-1",
        browse_id: null,
        directory_ref: null,
      },
    ]);
  });

  it("clears a selected folder when its machine disconnects", async () => {
    const replacement = {
      ...MACHINE,
      machine_id: "machine-2",
      connection_id: "connection-2",
      display_name: "Laptop",
    };
    useCopilotUIStore.setState({
      newChatExecutionTarget: {
        kind: "local",
        machineID: "machine-1",
        machineLabel: "Workstation",
        connectionID: "connection-1",
        browseID: "browse-1",
        directoryRef: "folder-1",
        displayPath: "C:\\Projects",
      },
      isExecutionTargetPickerOpen: true,
    });
    mockMachines([replacement]);
    server.use(
      http.post(
        "http://localhost:3000/api/proxy/api/copilot/executors/machine-2/directories",
        () =>
          HttpResponse.json({
            connection_id: "connection-2",
            browse_id: "browse-2",
            current: null,
            parent_ref: null,
            entries: [],
            next_cursor: null,
            truncated: false,
            expires_at: 1,
          }),
      ),
    );

    render(<ExecutionTargetPicker />);

    await waitFor(() => {
      expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual({
        kind: "local",
        machineID: "machine-2",
        machineLabel: "Laptop",
        connectionID: "connection-2",
        browseID: null,
        directoryRef: null,
        displayPath: null,
      });
    });
    await waitFor(() => {
      expect(useCopilotUIStore.getState().executionTargetError).toMatch(
        /offline/i,
      );
    });
  });
});

async function chooseLocalPC(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    screen.getByRole("button", { name: "Execution target: Cloud" }),
  );
  await user.click(
    await screen.findByRole("menuitemradio", { name: /Local PC/i }),
  );
  await waitFor(() => {
    expect(
      screen.getByRole("dialog", {
        name: /Choose a Folder on Your Local PC/i,
      }),
    ).toBeDefined();
  });
}
