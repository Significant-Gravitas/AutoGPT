import { truncate } from "../../tools/GenericTool/helpers";

interface AskMeta {
  ask: string;
  // The argument the headline names, first match wins; the card omits it from its fields.
  subject?: string[];
}

// Every tool the action gate can hold, as the user would ask for it.
export const TOOL_ASK: Record<string, AskMeta> = {
  bash_exec: { ask: "Run a command in the sandbox" },
  browser_act: { ask: "Act on the open page" },
  post_to_chat_platform: { ask: "Post a message" },
  edit_chat_platform_message: { ask: "Edit a posted message" },
  create_agent: { ask: "Create an agent" },
  customize_agent: { ask: "Customize an agent" },
  edit_agent: { ask: "Edit an agent" },
  fix_agent_graph: { ask: "Fix an agent" },
  enter_agent_building_mode: { ask: "Start building an agent" },
  create_folder: { ask: "Create folder", subject: ["name"] },
  update_folder: { ask: "Update folder", subject: ["name"] },
  move_folder: { ask: "Move a folder" },
  delete_folder: { ask: "Delete a folder" },
  move_agents_to_folder: { ask: "Move agents into a folder" },
  update_preset: { ask: "Update preset", subject: ["name"] },
  delete_preset: { ask: "Delete a preset" },
  schedule_followup: { ask: "Schedule a follow-up", subject: ["name"] },
  schedule_routine: { ask: "Schedule a routine", subject: ["title"] },
  pause_schedule: { ask: "Pause a schedule" },
  resume_schedule: { ask: "Resume a schedule" },
  delete_schedule: { ask: "Delete a schedule" },
  setup_agent_webhook_trigger: { ask: "Set up the trigger", subject: ["name"] },
  create_feature_request: { ask: "File a feature request", subject: ["title"] },
  hire_expert: { ask: "Hire", subject: ["name"] },
  raise_expert: { ask: "Create teammate", subject: ["name"] },
  update_expert: { ask: "Update teammate", subject: ["name"] },
  update_expert_soul: { ask: "Change how this teammate works" },
  confirm_expert_change: { ask: "Confirm the team change" },
  confirm_expert_soul_update: { ask: "Confirm the teammate change" },
  install_expert_workflow: { ask: "Give a teammate a workflow" },
  remove_expert_workflow: { ask: "Take a workflow from a teammate" },
  grant_expert_credential: { ask: "Give a teammate an account" },
  revoke_expert_credential: { ask: "Take an account from a teammate" },
  delegate_to_expert: { ask: "Hand a task to a teammate" },
  handoff_to_expert: { ask: "Hand this chat to a teammate" },
  message_session: { ask: "Message another chat" },
  run_sub_session: { ask: "Start a subtask" },
  delete_skill: { ask: "Delete skill", subject: ["name"] },
  delete_workspace_file: { ask: "Delete file", subject: ["path"] },
  memory_forget_confirm: { ask: "Forget memories" },
};

export interface AskLabel {
  ask: string;
  object: string | null;
  // Argument keys the headline already shows.
  shownKeys: string[];
}

export function getAskLabel(
  toolName: string,
  input: Record<string, unknown>,
): AskLabel | null {
  const meta = TOOL_ASK[toolName];
  if (!meta) return null;
  for (const key of meta.subject ?? []) {
    const value = input[key];
    if (typeof value === "string" && value.trim()) {
      return {
        ask: meta.ask,
        object: truncate(value.trim(), 60),
        shownKeys: [key],
      };
    }
  }
  return { ask: meta.ask, object: null, shownKeys: [] };
}

export function askText(label: AskLabel): string {
  return label.object ? `${label.ask} "${label.object}"` : label.ask;
}
