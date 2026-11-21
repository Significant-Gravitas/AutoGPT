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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-1" } }),
}));

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
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
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

  it("retries a failed page without reloading the folder", async () => {
    const user = userEvent.setup();
    mockMachines();
    const requests: Array<Record<string, unknown>> = [];
    let failNextPage = true;
    const home = {
      directory_ref: "home-ref",
      name: "Home",
      path: "C:\\Users\\Ada",
    };
    server.use(
      http.post(DIRECTORIES_URL, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        requests.push(body);
        if (body.cursor === "next-page") {
          if (failNextPage) {
            failNextPage = false;
            return HttpResponse.json({ detail: "busy" }, { status: 503 });
          }
          return HttpResponse.json({
            connection_id: "connection-1",
            browse_id: "browse-1",
            current: home,
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
            current: home,
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
          entries: [home],
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
    await user.click(await screen.findByRole("button", { name: "Retry" }));

    expect(
      await screen.findByRole("button", { name: "Second Page" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "First Page" })).toBeDefined();
    expect(requests.slice(-2)).toEqual([
      expect.objectContaining({ cursor: "next-page" }),
      expect.objectContaining({ cursor: "next-page" }),
    ]);
  });

  it("teaches setup and keeps polling when no computer is connected", async () => {
    const user = userEvent.setup();
    mockMachines([]);

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);

    expect(await screen.findByText("Connect a Computer")).toBeDefined();
    expect(
      screen.getByText(
        /^pipx install git\+https:\/\/github\.com\/Significant-Gravitas\/autogpt-local-executor\.git@[a-f0-9]{40}$/,
      ),
    ).toBeDefined();
    expect(
      screen.getByText(
        /autogpt-shim --platform-url .* --platform-oauth-url .* auth/,
      ),
    ).toBeDefined();
    expect(screen.getByText("autogpt-shim install")).toBeDefined();
    expect(
      screen.getByText(/even when this chat is open on another device/i),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Check Again" })).toBeDefined();
  });

  it("shows the setup card as waiting, with nothing on it that looks like success", async () => {
    const user = userEvent.setup();
    mockMachines([]);

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);

    const waiting = await screen.findByText(
      "Waiting for a signed-in executor…",
    );
    const card = screen.getByRole("region", { name: "Connect a Computer" });
    expect(inASuccessColour(card)).toEqual([]);
    expect(
      waiting.querySelector('[data-slot="loading-spinner"]'),
    ).not.toBeNull();
  });

  it("invalidates the folder when a paginated response changes its binding", async () => {
    const user = userEvent.setup();
    mockMachines();
    server.use(
      http.post(DIRECTORIES_URL, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        const home = {
          directory_ref: "home-ref",
          name: "Home",
          path: "C:\\Users\\Ada",
        };
        return HttpResponse.json({
          connection_id: "connection-1",
          browse_id: body.cursor ? "unexpected-browse" : "browse-1",
          current: body.directory_ref ? home : null,
          parent_ref: null,
          entries: body.directory_ref
            ? [{ ...home, directory_ref: "child-ref", name: "Child" }]
            : [home],
          next_cursor: body.directory_ref ? "next-page" : null,
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

    await screen.findAllByText(/folder view changed/i);
    expect(
      (
        screen.getByRole("button", {
          name: "Use This Folder",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.queryByText("C:\\Users\\Ada")).toBeNull();
  });

  it("invalidates the folder when opening one returns a different folder", async () => {
    const user = userEvent.setup();
    mockMachines();
    server.use(
      http.post(DIRECTORIES_URL, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        const home = {
          directory_ref: "home-ref",
          name: "Home",
          path: "C:\\Users\\Ada",
        };
        const other = {
          directory_ref: "other-ref",
          name: "Other",
          path: "C:\\Users\\Other",
        };
        return HttpResponse.json({
          connection_id: "connection-1",
          browse_id: "browse-1",
          current: body.directory_ref ? other : null,
          parent_ref: null,
          entries: body.directory_ref ? [] : [home],
          next_cursor: null,
          truncated: false,
          expires_at: 1,
        });
      }),
    );

    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);
    await user.click(await screen.findByRole("button", { name: "Home" }));

    await screen.findAllByText(/folder view changed/i);
    expect(
      (
        screen.getByRole("button", {
          name: "Use This Folder",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.queryByText("C:\\Users\\Other")).toBeNull();
  });

  it("preserves a selected folder while the first machine check is pending", async () => {
    const selectedTarget = {
      kind: "local" as const,
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "connection-1",
      browseID: "browse-1",
      directoryRef: "folder-1",
      displayPath: "C:\\Projects",
    };
    // The computers are checked only while the folder dialog is open.
    useCopilotUIStore.setState({
      newChatExecutionTarget: selectedTarget,
      isExecutionTargetPickerOpen: true,
    });
    const checkStarted = vi.fn();
    const heldChecks: Array<() => void> = [];
    let checksReleased = false;
    server.use(
      http.get(EXECUTORS_URL, async () => {
        checkStarted();
        if (!checksReleased) {
          await new Promise<void>((resolve) => heldChecks.push(resolve));
        }
        return HttpResponse.json({ executors: [MACHINE] });
      }),
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

    try {
      await waitFor(() => expect(checkStarted).toHaveBeenCalled());
      expect(
        await screen.findByText("Looking for connected computers…"),
      ).toBeDefined();
      expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual(
        selectedTarget,
      );
    } finally {
      // Frees every held check; the 3 s polls after this answer at once.
      checksReleased = true;
      heldChecks.splice(0).forEach((release) => release());
    }

    // The folders load once the computers have, so the check is over.
    expect(
      await screen.findByText("This folder has no subfolders."),
    ).toBeDefined();
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual(
      selectedTarget,
    );
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

  it("clears a previously confirmed folder after a stale browse response", async () => {
    const user = userEvent.setup();
    mockMachines();
    let stale = false;
    server.use(
      http.post(DIRECTORIES_URL, () =>
        stale
          ? HttpResponse.json({ detail: "Browse expired" }, { status: 409 })
          : HttpResponse.json({
              connection_id: "connection-1",
              browse_id: "browse-1",
              current: {
                directory_ref: "home-ref",
                name: "Home",
                path: "C:\\Users\\Ada",
              },
              parent_ref: null,
              entries: [],
              truncated: false,
              expires_at: 1,
            }),
      ),
    );
    render(<ExecutionTargetPicker />);
    await chooseLocalPC(user);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Use This Folder" }),
      ).toHaveProperty("disabled", false);
    });
    await user.click(screen.getByRole("button", { name: "Use This Folder" }));
    stale = true;
    await user.click(screen.getByRole("button", { name: /Local folder:/ }));

    await screen.findAllByText(/computer or folder changed/i);
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toMatchObject({
      kind: "local",
      machineID: "machine-1",
      connectionID: "connection-1",
      browseID: null,
      directoryRef: null,
      displayPath: null,
    });
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

// The colour roles are the design system's vocabulary: a success role on an
// element is what makes it look like success.
function inASuccessColour(root: HTMLElement) {
  return Array.from(root.querySelectorAll("*")).filter((element) =>
    /\bsuccess\b/.test(element.getAttribute("class") ?? ""),
  );
}

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
