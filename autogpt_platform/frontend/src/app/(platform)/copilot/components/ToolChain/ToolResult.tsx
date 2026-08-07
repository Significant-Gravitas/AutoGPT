"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { useContext } from "react";
import { PendingQuestionsContext } from "../QuestionDock/PendingQuestionsContext";
import { QuestionsForm } from "../QuestionDock/QuestionDock";
import { SetupRequirementsCard } from "../SetupRequirementsCard/SetupRequirementsCard";
import { MCPSetupCard } from "../../tools/RunMCPTool/components/MCPSetupCard/MCPSetupCard";
import {
  AgentListCard,
  AgentPreviewCard,
  AgentSavedCard,
  SubSessionCard,
} from "./AgentCards";
import { BlockListCard, BlockOutputCard } from "./BlockCards";
import { ExecutionCard } from "./ExecutionCard";
import { FileDiff } from "./FileDiff";
import { isDiffText } from "./fileDiffHelpers";
import type { ChainRow } from "./helpers";
import {
  FixResultCard,
  PlanSteps,
  QuestionsCard,
  SkillCard,
  SuggestedGoalCard,
  TriggerSetupCard,
  ValidationCard,
} from "./InfoCards";
import { extractClarifyingQuestions } from "../../tools/clarifying-questions";
import {
  DocsList,
  FeatureRequestList,
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

function setupRequirementsCard(row: ChainRow, output: Record<string, unknown>) {
  const setupInfo = asObject(output.setup_info);
  if (!setupInfo) return null;
  const setupOutput = output as unknown as SetupRequirementsResponse;

  if (row.tool === "run_mcp_tool") {
    return (
      <MCPSetupCard
        output={setupOutput}
        retryInstruction="I've connected the MCP server credentials. Please retry run_mcp_tool with the same server URL and arguments."
      />
    );
  }

  if (row.tool === "setup_agent_webhook_trigger") {
    return (
      <SetupRequirementsCard
        output={setupOutput}
        inputsMode="trigger"
        credentialsLabel="Account"
      />
    );
  }

  if (row.tool === "run_agent" || row.tool === "schedule_agent") {
    return <SetupRequirementsCard output={setupOutput} inputsMode="preview" />;
  }

  return (
    <SetupRequirementsCard
      output={setupOutput}
      credentialsLabel={
        row.tool === "connect_integration"
          ? `${str(setupInfo, "agent_name") ?? "Integration"} credentials`
          : undefined
      }
      retryInstruction={
        row.tool === "connect_integration"
          ? "I've connected my account. Please continue."
          : undefined
      }
    />
  );
}

function toolCard(row: ChainRow, output: Record<string, unknown> | null) {
  const input = asObject(row.input);

  if (output) {
    const setupCard = setupRequirementsCard(row, output);
    if (setupCard) return setupCard;
    const questions = asItems(output.questions);
    if (questions) return <QuestionsCard questions={questions} />;
  }

  switch (row.tool) {
    case "run_agent":
    case "schedule_agent": {
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
      const agent = output && asObject(output.agent);
      if (agent) return <AgentListCard agents={[agent]} />;
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
      if (output && str(output, "suggested_goal")) {
        return <SuggestedGoalCard output={output} />;
      }
      const name = output && str(output, "agent_name", "name", "graph_name");
      if (!output || !name) return null;
      return str(output, "library_agent_link", "agent_page_link") ? (
        <AgentSavedCard output={output} />
      ) : (
        <AgentPreviewCard output={output} />
      );
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
      const block = asObject(output.block);
      if (block) return <BlockListCard blocks={[block]} />;
      if (str(output, "block_name", "block_id"))
        return <BlockOutputCard output={output} />;
      return null;
    }
    case "connect_integration":
      return null;
    case "decompose_goal": {
      const steps = output && asItems(output.steps);
      return steps ? <PlanSteps steps={steps} /> : null;
    }
    case "validate_agent_graph":
      return output ? <ValidationCard output={output} /> : null;
    case "fix_agent_graph":
      return output ? <FixResultCard output={output} /> : null;
    // Interactive answering lives in the QuestionDock above the chat input;
    // the chain row keeps a read-only record of what was asked.
    case "ask_question": {
      const questions = extractClarifyingQuestions(row);
      return questions.length > 0 ? (
        <QuestionsCard
          questions={questions as unknown as Record<string, unknown>[]}
        />
      ) : null;
    }
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
    case "setup_agent_webhook_trigger":
      return output &&
        (str(output, "webhook_url", "ingress_url") ||
          output.manual_setup_required === true) ? (
        <TriggerSetupCard output={output} />
      ) : null;
    case "search_feature_requests": {
      const results = output && asItems(output.results);
      return results ? <FeatureRequestList results={results} /> : null;
    }
    case "create_feature_request":
      return output
        ? (linkCard(output, input) ?? <KeyValueList value={output} />)
        : null;
    case "run_mcp_tool":
      return output && "result" in output ? (
        <KeyValueList value={output.result} />
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

interface Props {
  row: ChainRow;
}

export function ToolResult({ row }: Props) {
  const output = asObject(row.output);
  const pendingQuestions = useContext(PendingQuestionsContext);

  // The latest unanswered clarifying questions render as an interactive
  // answer form right on their chain row (older ones keep the read-only
  // card from toolCard).
  if (pendingQuestions?.callIds.includes(row.key)) {
    return (
      <QuestionsForm
        key={pendingQuestions.dockId}
        dockId={pendingQuestions.dockId}
        questions={pendingQuestions.questions}
      />
    );
  }

  if (!output && isDiffText(row.output)) {
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
