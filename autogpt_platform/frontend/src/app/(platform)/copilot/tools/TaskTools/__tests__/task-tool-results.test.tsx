import userEvent from "@testing-library/user-event";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";
import type { ChainRow } from "../../../components/ToolChain/helpers";
import { ToolResult } from "../../../components/ToolChain/ToolResult";
import { useWorkAreaStore } from "../../../workArea";

function toolRow(tool: string, output: unknown): ChainRow {
  return {
    key: `call-${tool}`,
    category: "todo",
    text: tool,
    state: "done",
    tool,
    input: {},
    output,
  } as unknown as ChainRow;
}

function taskInfo(extra: Record<string, unknown> = {}) {
  return {
    id: "task-7",
    title: "Find 25 look-alike design partners",
    description: "Keep it friendly.",
    status: "IN_PROGRESS",
    owner_expert_id: "expert-maria",
    created_by: "OTTO",
    session_ids: ["sess-1"],
    updated_at: "2026-11-19T09:00:00Z",
    attempt: 1,
    ...extra,
  };
}

const STARTED = {
  type: "task_started",
  message:
    "Started 'Find 25 look-alike design partners' in the background; tell the user.",
  session_id: "sess-1",
  task: taskInfo(),
  in_background: true,
  worker_session_id: "worker-9",
  worker_session_link: "/copilot?sessionId=worker-9",
  background_task_id: "bg-3",
  start_at: null,
};

const INTERNALS = [
  /task-7/,
  /worker-9/,
  /bg-3/,
  /sessionId=/,
  /In background/i,
  /Worker session/i,
  /Background task/i,
  /\btrue\b/,
  /\{"id"/,
];

function expectNoInternals() {
  const text = document.body.textContent ?? "";
  for (const pattern of INTERNALS) expect(text).not.toMatch(pattern);
}

beforeEach(() => {
  useWorkAreaStore.getState().reset();
  server.use(
    getListExpertIdentitiesMockHandler([
      {
        id: "expert-maria",
        name: "Maria",
        avatar_url: null,
        role: "Marketing Strategist",
        is_archived: false,
      },
    ]),
  );
});

describe("task steps in the chat, expanded", () => {
  it("shows a started task as its title, owner and status, with Open", async () => {
    render(<ToolResult row={toolRow("start_task", STARTED)} />);

    expect(
      screen.getByText("Find 25 look-alike design partners"),
    ).toBeDefined();
    expect(await screen.findByText(/Maria/)).toBeDefined();
    expect(screen.getByText(/Working/)).toBeDefined();
    expectNoInternals();

    await userEvent.click(
      screen.getByRole("button", {
        name: "Open task Find 25 look-alike design partners",
      }),
    );
    const state = useWorkAreaStore.getState();
    expect(state.activeKey).toBe("task:task-7");
    expect(state.tabs).toContainEqual(
      expect.objectContaining({ kind: "task", taskID: "task-7" }),
    );
  });

  it("says when a task waiting for its start time starts", () => {
    const later = {
      ...STARTED,
      in_background: false,
      start_at: "2099-01-05T09:00:00Z",
      task: taskInfo({
        status: "TODO",
        owner_expert_id: null,
        start_at: "2099-01-05T09:00:00Z",
      }),
    };
    render(<ToolResult row={toolRow("start_task", later)} />);

    expect(screen.getByText(/Otto/)).toBeDefined();
    expect(screen.getByText(/Starts /)).toBeDefined();
    expectNoInternals();
  });

  it("reads an old stored result that arrives as JSON text", () => {
    render(
      <ToolResult row={toolRow("start_later", JSON.stringify(STARTED))} />,
    );

    expect(
      screen.getByRole("button", {
        name: "Open task Find 25 look-alike design partners",
      }),
    ).toBeDefined();
    expectNoInternals();
  });

  it.each(["create_task", "update_task", "get_task"])(
    "draws %s's task the same way",
    (tool) => {
      const type =
        tool === "create_task"
          ? "task_created"
          : tool === "update_task"
            ? "task_updated"
            : "task_details";
      render(
        <ToolResult
          row={toolRow(tool, {
            type,
            message: "ok",
            task: taskInfo({ status: "DONE" }),
          })}
        />,
      );
      expect(screen.getByText(/Done/)).toBeDefined();
      expect(
        screen.getByRole("button", {
          name: "Open task Find 25 look-alike design partners",
        }),
      ).toBeDefined();
      expectNoInternals();
    },
  );

  it("says Queued for a started task waiting for a free slot", async () => {
    render(
      <ToolResult
        row={toolRow("start_task", {
          ...STARTED,
          task: taskInfo({ is_queued: true }),
        })}
      />,
    );
    expect(await screen.findByText(/Maria · Queued/)).toBeDefined();
    expect(screen.queryByText(/Working/)).toBeNull();
  });

  it("lists tasks one line each", () => {
    render(
      <ToolResult
        row={toolRow("list_tasks", {
          type: "task_list",
          message: "2 tasks",
          count: 2,
          tasks: [
            taskInfo(),
            taskInfo({
              id: "task-8",
              title: "Draft the launch post",
              status: "FAILED",
            }),
          ],
        })}
      />,
    );

    expect(screen.getAllByRole("button", { name: /^Open task / })).toHaveLength(
      2,
    );
    expect(screen.getByText("Draft the launch post")).toBeDefined();
    expect(screen.getByText(/Failed/)).toBeDefined();
    expect(document.body.textContent).not.toMatch(/task-8|count/);
  });

  it("says a task was archived or its result saved, without ids", () => {
    const { unmount } = render(
      <ToolResult
        row={toolRow("archive_task", {
          type: "task_archived",
          message: "Archived the task.",
          task_id: "task-7",
        })}
      />,
    );
    expect(screen.getByText("Task archived.")).toBeDefined();
    expectNoInternals();
    unmount();

    render(
      <ToolResult
        row={toolRow("finish_task", {
          type: "task_finished",
          message: "Saved. End your turn now.",
          document: "Design partners.md",
        })}
      />,
    );
    expect(
      screen.getByText("Saved the result in Design partners.md."),
    ).toBeDefined();
    expect(document.body.textContent).not.toMatch(/End your turn/);
  });

  it("says a result saved as not done, with the reason", () => {
    render(
      <ToolResult
        row={{
          ...toolRow("finish_task", {
            type: "task_finished",
            message:
              "Saved on the task as not done: it shows as failed with your reason.",
            document: null,
          }),
          input: {
            summary: "No web search was available.",
            outcome: "blocked",
          },
        }}
      />,
    );
    expect(
      screen.getByText("Saved as not done: No web search was available."),
    ).toBeDefined();
    expect(screen.queryByText(/Saved the result/)).toBeNull();
  });

  it("keeps an error as the tool said it", () => {
    render(
      <ToolResult
        row={toolRow("start_task", {
          type: "error",
          message: "Could not start it.",
        })}
      />,
    );
    expect(screen.queryByRole("button", { name: /Open task/ })).toBeNull();
    expect(screen.getByText(/Could not start it\./)).toBeDefined();
  });
});
