"use client";

import { AgentListCard, AgentSavedCard, SubSessionCard } from "./AgentCards";
import { BlockListCard, BlockOutputCard } from "./BlockCards";
import { ExecutionCard } from "./ExecutionCard";
import { FileDiff, isDiffText } from "./FileDiff";
import type { ChainRow } from "./helpers";
import {
  FixResultCard,
  PlanSteps,
  SetupCard,
  SkillCard,
  ValidationCard,
} from "./InfoCards";
import { QuestionRowForm } from "./QuestionRowForm";
import {
  DocsList,
  FileList,
  FolderList,
  ScheduleCreatedCard,
  ScheduleList,
} from "./ListCards";
import {
  CARD,
  ChipList,
  HALF,
  LinkCard,
  StatCard,
  StatusCard,
  StatusPill,
} from "./ResultCards";
import {
  asItems,
  asObject,
  dictToOutputItems,
  humanizeKey,
  str,
  stripBaseFields,
} from "./resultHelpers";
import {
  FileCard,
  KeyValueList,
  OutputList,
  SearchResults,
  Terminal,
  TodoList,
} from "./ToolResultViews";

const BOOLEAN_LABELS: Record<string, string> = {
  ok: "Done",
  success: "Done",
};

function shapeCard(obj: Record<string, unknown>) {
  const entries = Object.entries(obj);
  if (entries.length !== 1) return null;
  const [key, value] = entries[0];

  if (typeof value === "boolean")
    return (
      <StatusCard ok={value} label={BOOLEAN_LABELS[key] ?? humanizeKey(key)} />
    );

  if (typeof value === "number")
    return <StatCard value={value} label={humanizeKey(key).toLowerCase()} />;

  if (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => typeof item === "string")
  )
    return <ChipList label={humanizeKey(key)} items={value as string[]} />;

  if (key === "status" && typeof value === "string")
    return (
      <div className={`${CARD} ${HALF} flex items-center gap-2 p-2.5`}>
        <span className="text-xs text-zinc-400">Status</span>
        <span className="ml-auto">
          <StatusPill status={value} />
        </span>
      </div>
    );

  return null;
}

function linkCard(
  output: Record<string, unknown>,
  input: Record<string, unknown> | null,
) {
  const url =
    str(output, "url", "ingress_url", "link", "webhook_url", "issue_url") ??
    (input ? str(input, "url", "server_url") : null);
  if (!url || !/^https?:\/\//.test(url)) return null;

  const { status, status_code, bytes, content_length } = output;
  const size =
    typeof bytes === "number"
      ? bytes
      : typeof content_length === "number"
        ? content_length
        : null;
  const code =
    typeof status === "number"
      ? status
      : typeof status_code === "number"
        ? status_code
        : null;
  const meta =
    size !== null
      ? `${(size / 1024).toFixed(1)} KB`
      : code !== null
        ? String(code)
        : undefined;

  return (
    <LinkCard
      url={url}
      title={str(output, "title", "issue_title") ?? undefined}
      meta={meta}
    />
  );
}

function chipStrings(value: unknown, key: string): string[] | null {
  const items = asItems(value);
  if (!items) return null;
  const labels = items
    .map((item) => str(item, key))
    .filter((label): label is string => !!label);
  return labels.length > 0 ? labels : null;
}

function toolCard(row: ChainRow, output: Record<string, unknown> | null) {
  const input = asObject(row.input);

  switch (row.tool) {
    case "run_agent": {
      const name =
        (output && str(output, "graph_name", "agent_name")) ??
        (input ? str(input, "username_agent_slug", "agent_name") : null);
      if (output && name && str(output, "execution_id")) {
        const graphID = str(output, "graph_id");
        const executionID = str(output, "execution_id");
        return (
          <ExecutionCard
            name={name}
            status={str(output, "status") ?? undefined}
            href={
              str(output, "library_agent_link") ??
              (graphID && executionID
                ? `/library/agents/${graphID}?activeTab=runs&activeItem=${executionID}`
                : undefined)
            }
          />
        );
      }
      const execution = output && asObject(output.execution);
      const items = execution && dictToOutputItems(execution.outputs);
      return items ? <OutputList items={items} /> : null;
    }
    case "view_agent_output": {
      const execution = output && asObject(output.execution);
      const items =
        (execution && dictToOutputItems(execution.outputs)) ??
        (output && asItems(output.outputs));
      return items ? <OutputList items={items} /> : null;
    }
    case "create_agent":
    case "customize_agent":
    case "edit_agent": {
      const name = output && str(output, "agent_name", "name", "graph_name");
      return output && name ? <AgentSavedCard output={output} /> : null;
    }
    case "run_sub_session":
    case "get_sub_session_result":
      return output && str(output, "status") ? (
        <SubSessionCard output={output} />
      ) : null;
    case "find_agent":
    case "find_library_agent": {
      const agents = output && asItems(output.agents);
      return agents ? <AgentListCard agents={agents} /> : null;
    }
    case "find_block": {
      const blocks = output && asItems(output.blocks);
      return blocks ? <BlockListCard blocks={blocks} /> : null;
    }
    case "run_block":
    case "continue_run_block": {
      if (!output) return null;
      if (str(output, "type") === "setup_requirements")
        return <SetupCard output={output} provider={null} />;
      const block = asObject(output.block);
      if (block) return <BlockListCard blocks={[block]} />;
      if (str(output, "block_name", "block_id"))
        return <BlockOutputCard output={output} />;
      return null;
    }
    case "connect_integration":
      return output && str(output, "type") === "setup_requirements" ? (
        <SetupCard
          output={output}
          provider={input ? str(input, "provider") : null}
        />
      ) : null;
    case "decompose_goal": {
      const steps = output && asItems(output.steps);
      return steps ? <PlanSteps steps={steps} /> : null;
    }
    case "validate_agent_graph":
      return output ? <ValidationCard output={output} /> : null;
    case "fix_agent_graph":
      return output ? <FixResultCard output={output} /> : null;
    case "ask_question":
      return <QuestionRowForm row={row} />;
    case "list_schedules": {
      const schedules = output && asItems(output.schedules);
      return schedules ? <ScheduleList schedules={schedules} /> : null;
    }
    case "schedule_followup":
      return output && str(output, "next_run_time") ? (
        <ScheduleCreatedCard output={output} />
      ) : null;
    case "list_folders": {
      const folders = output && asItems(output.folders);
      return folders ? <FolderList folders={folders} /> : null;
    }
    case "create_folder":
    case "update_folder":
    case "move_folder": {
      const folder = output && asObject(output.folder);
      return folder ? <FolderList folders={[folder]} /> : null;
    }
    case "list_workspace_files": {
      const files = output && asItems(output.files);
      return files ? <FileList files={files} /> : null;
    }
    case "search_docs": {
      const results = output && asItems(output.results);
      return results ? <DocsList results={results} /> : null;
    }
    case "get_doc_page":
      return output && str(output, "title") ? (
        <DocsList results={[output]} />
      ) : null;
    case "store_skill":
    case "read_skill":
    case "delete_skill":
      return output ? <SkillCard output={output} /> : null;
    case "list_skills": {
      const names = output && chipStrings(output.skills, "name");
      return names ? <ChipList label="Skills" items={names} /> : null;
    }
    case "list_chat_platform_channels": {
      const names = output && chipStrings(output.channels, "name");
      return names ? (
        <ChipList label="Channels" items={names.map((name) => `#${name}`)} />
      ) : null;
    }
    case "web_search": {
      const items = output && asItems(output.results);
      const answer = output && str(output, "answer");
      if (items) return <SearchResults items={items} answer={answer} />;
      if (answer) return <KeyValueList value={answer} />;
      return null;
    }
    case "bash_exec":
      return <Terminal row={row} />;
    case "TodoWrite":
      return <TodoList row={row} />;
    case "read_workspace_file":
    case "write_workspace_file":
    case "delete_workspace_file":
    case "Read":
    case "Write":
      return <FileCard row={row} />;
  }
  return null;
}

export function ToolResult({ row }: { row: ChainRow }) {
  if (isDiffText(row.output)) {
    const input = asObject(row.input);
    return (
      <FileDiff
        file={
          input ? (str(input, "file_path", "path") ?? undefined) : undefined
        }
        diff={row.output}
      />
    );
  }

  const output = asObject(row.output);

  const card = toolCard(row, output);
  if (card) return card;

  if (!output) return <KeyValueList value={row.output} />;

  const data = stripBaseFields(output);

  const outputs = asItems(data.outputs) ?? dictToOutputItems(data.outputs);
  if (outputs) return <OutputList items={outputs} />;

  if (Object.keys(data).length === 0)
    return <KeyValueList value={str(output, "message") ?? ""} />;

  return (
    linkCard(data, asObject(row.input)) ??
    shapeCard(data) ?? <KeyValueList value={data} />
  );
}
