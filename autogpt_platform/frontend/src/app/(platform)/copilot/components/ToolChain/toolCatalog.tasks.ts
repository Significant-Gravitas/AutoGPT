import { quoted, type ToolMeta } from "./toolCatalog.shared";

/** Tasks v1 tools. Plain step labels only: what they return is for Otto,
 *  so it stays folded under the chain. */
export const TASK_TOOL_CATALOG: Record<string, ToolMeta> = {
  create_task: {
    category: "todo",
    running: "Creating task",
    done: "Created task",
    subject: (input) => quoted(input, "title"),
  },
  list_my_tasks: {
    category: "todo",
    running: "Checking your tasks",
    done: "Checked your tasks",
    empty: "none open",
  },
  update_task_status: {
    category: "todo",
    running: "Updating the task",
    done: "Updated the task",
  },
  update_task: {
    category: "todo",
    running: "Updating the task",
    done: "Updated the task",
  },
  comment_on_task: {
    category: "todo",
    running: "Adding a note to the task",
    done: "Added a note to the task",
  },
  request_input: {
    category: "question",
    running: "Asking you for input",
    done: "Asked you for input",
  },
  ask_user: {
    category: "question",
    running: "Asking you a question",
    done: "Asked you a question",
  },
  link_artifact: {
    category: "todo",
    running: "Attaching the result",
    done: "Attached the result",
  },
  add_evidence: {
    category: "todo",
    running: "Attaching the result",
    done: "Attached the result",
  },
  propose_plan: {
    category: "plan",
    running: "Drafting a plan",
    done: "Drafted a plan",
  },
  confirm_task_plan: {
    category: "plan",
    running: "Starting the plan",
    done: "Started the plan",
  },
};
