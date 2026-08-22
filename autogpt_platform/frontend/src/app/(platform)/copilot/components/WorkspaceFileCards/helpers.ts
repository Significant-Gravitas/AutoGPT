import type { UIMessage } from "ai";
import { asObject, str } from "../ToolChain/resultHelpers";

export interface SessionRun {
  executionId: string;
  /** Null while a run is in flight and was started by library-agent id — the
   *  tool input carries no name, so the row resolves one itself. */
  name: string | null;
  libraryAgentId: string | null;
  status: string | null;
  /** Only the waited (synchronous) envelope carries one — a fire-and-forget
   *  run answers before the executor has stamped a start time. */
  startedAt: string | null;
  href: string | null;
}

export interface SessionSchedule {
  scheduleId: string;
  name: string;
  nextRunTime: string | null;
  isRecurring: boolean;
  /** What actually fires: the follow-up prompt, or the agent being run. Only
   *  the tool input has it — the created-schedule envelope is id + time. */
  detail: string | null;
  cron: string | null;
  timezone: string | null;
}

const RUN_TOOLS = new Set(["tool-run_agent", "tool-schedule_agent"]);

/**
 * Runs triggered and schedules created in this chat, mined from the session's
 * tool outputs — there is no session-scoped REST list for either, the chat
 * transcript is the source of truth.
 */
export function getSessionActivity(messages: UIMessage[]): {
  runs: SessionRun[];
  schedules: SessionSchedule[];
} {
  const runs = new Map<string, SessionRun>();
  const schedules = new Map<string, SessionSchedule>();
  const deletedScheduleIds = new Set<string>();

  for (const message of messages) {
    for (const part of message.parts) {
      if (!part.type.startsWith("tool-")) continue;
      const output = asObject((part as { output?: unknown }).output);
      const toolInput = asObject((part as { input?: unknown }).input);

      // A waited run has no output until the agent finishes — surface it
      // from the tool input as an in-flight row so the card shows the run
      // while it executes.
      if (!output && RUN_TOOLS.has(part.type)) {
        const state = (part as { state?: string }).state;
        if (state !== "input-streaming" && state !== "input-available")
          continue;
        const input = toolInput;
        if (input && str(input, "cron", "schedule_name")) continue;
        const key =
          (part as { toolCallId?: string }).toolCallId ??
          `pending-${runs.size}`;
        // ``run_agent`` takes a slug or a library-agent id, never a name.
        // The slug's trailing segment is the agent, so it reads fine once
        // de-slugified; an id-launched run has to be looked up by the row.
        const slug = input ? str(input, "username_agent_slug") : null;
        runs.set(key, {
          executionId: key,
          name: slug ? deslugify(slug) : null,
          libraryAgentId: input ? str(input, "library_agent_id") : null,
          status: "RUNNING",
          startedAt: null,
          href: null,
        });
        continue;
      }
      if (!output) continue;

      if (RUN_TOOLS.has(part.type)) {
        const name = str(output, "graph_name", "agent_name");
        // Async runs answer with an execution_started envelope
        // (execution_id at the top level); synchronous waited runs answer
        // with agent_output, nesting the id under `execution`.
        const execution = asObject(output.execution);
        const executionId =
          str(output, "execution_id") ??
          (execution ? str(execution, "execution_id") : null);
        const status =
          str(output, "status") ??
          (execution ? str(execution, "status") : null);
        const graphId = str(output, "graph_id", "agent_id");
        if (executionId) {
          // Agent schedules ride the same execution_started envelope with a
          // SCHEDULED status and the schedule id in execution_id.
          if (status === "SCHEDULED") {
            schedules.set(executionId, {
              scheduleId: executionId,
              name:
                (toolInput ? str(toolInput, "schedule_name") : null) ??
                name ??
                "Schedule",
              nextRunTime: str(output, "next_run_time"),
              isRecurring: true,
              detail: name ? `Runs ${name}` : null,
              cron: toolInput ? str(toolInput, "cron") : null,
              timezone: toolInput ? str(toolInput, "timezone") : null,
            });
          } else {
            runs.set(executionId, {
              executionId,
              name,
              libraryAgentId: str(output, "library_agent_id"),
              status,
              startedAt:
                str(output, "started_at") ??
                (execution ? str(execution, "started_at") : null),
              href:
                str(output, "library_agent_link") ??
                (graphId
                  ? `/library/agents/${graphId}?activeTab=runs&activeItem=${executionId}`
                  : null),
            });
          }
        }
      }

      const scheduleId = str(output, "schedule_id");
      if (output.type === "schedule_deleted" && scheduleId) {
        deletedScheduleIds.add(scheduleId);
        continue;
      }
      const isCreatedSchedule =
        output.type === "schedule_created" ||
        (part.type === "tool-schedule_followup" &&
          !!str(output, "next_run_time"));
      if (isCreatedSchedule && scheduleId) {
        const cron =
          str(output, "cron") ?? (toolInput ? str(toolInput, "cron") : null);
        schedules.set(scheduleId, {
          scheduleId,
          name:
            str(output, "name", "schedule_name", "graph_name") ??
            (part.type === "tool-schedule_followup" ? "Follow-up" : "Schedule"),
          nextRunTime: str(output, "next_run_time"),
          isRecurring: !!cron || output.is_recurring === true,
          // A follow-up's payload is the message it will send — the single
          // most useful thing to show, and only the input has it.
          detail:
            str(output, "message") ??
            (toolInput ? str(toolInput, "message", "prompt") : null),
          cron,
          timezone: toolInput ? str(toolInput, "timezone") : null,
        });
      }
    }
  }

  for (const id of deletedScheduleIds) schedules.delete(id);

  // Newest first — later messages sit later in the transcript.
  return {
    runs: [...runs.values()].reverse(),
    schedules: [...schedules.values()].reverse(),
  };
}

/** "creator/daily-digest" → "daily digest". */
function deslugify(slug: string): string {
  const agent = slug.split("/").pop() ?? slug;
  return agent.replace(/[-_]+/g, " ").trim();
}
