import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { reactCompilerSources } from "../../../../../next.config.mjs";

// Vitest runs these files uncompiled, so a test that renders them cannot see
// what the compiler caches. This compiles them with the pipeline Next uses
// and checks the values that used to be hand-memoised are still cached: each
// one reaches a prop, a context or a hook dependency, where a new object on
// every render means extra renders, or an effect that re-runs.
const require = createRequire(import.meta.url);

const babel = require("next/dist/compiled/babel/core") as {
  transformSync: (code: string, options: object) => { code: string } | null;
};
const syntaxJsx = require("next/dist/compiled/babel/plugin-syntax-jsx");
const presetTypescript = require("next/dist/compiled/babel/preset-typescript");
const reactCompiler = require("babel-plugin-react-compiler");

const FRONTEND_ROOT = path.resolve(__dirname, "../../../../..");
const COPILOT = "src/app/(platform)/copilot";
const ORGANISMS = "src/components/organisms";

/** File → source snippets of the compiled output that must be cached. Most
 *  were hand-memoised before the compiler; a few still are (see the comment
 *  at each), because without the memo the compiler could not cache them
 *  across a later hook call. */
const CACHED: Record<string, string[]> = {
  [`${COPILOT}/useChatSession.ts`]: ["resolveSessionDryRun("],
  [`${COPILOT}/useCopilotPage.ts`]: [
    "resolveExpertIdentity(",
    "mergeTurnStats(",
  ],
  [`${COPILOT}/useCopilotRuntimeStream.ts`]: ["withLatestUser("],
  [`${COPILOT}/useCopilotPendingChips.ts`]: [
    "queue.map(",
    "function queueMessage(",
  ],
  [`${COPILOT}/components/ChatContainer/ChatContainer.tsx`]: [
    "<AutopilotModeSelector",
    "onStart: toggleVoice",
    "<VoiceModeBar",
  ],
  [`${COPILOT}/components/ChatInput/useVoiceRecording.ts`]: [
    "function startRecording(",
    "function toggleRecording(",
    "function handleKeyDown(",
  ],
  [`${COPILOT}/components/ChatMessagesContainer/ChatMessagesContainer.tsx`]: [
    "buildMessageIndexById(",
    "extractReviewTargets(",
    "hasTrailingErrorMarker(",
    "<TailSpacer",
  ],
  [`${COPILOT}/components/ChatMessagesContainer/components/PendingAnswerContexts.tsx`]:
    [
      "getPendingQuestions(messages)",
      "getAnsweredQuestions(messages)",
      "getHeldOutcomesOrNone(messages)",
    ],
  [`${COPILOT}/components/ChatMessagesContainer/components/useTTSButton.ts`]: [
    "stripMarkdownForSpeech(",
  ],
  [`${COPILOT}/components/ChatMessagesContainer/components/MessagePartRenderer.tsx`]:
    ["extractWorkspaceArtifacts(", "resolveWorkspaceUrls("],
  [`${COPILOT}/components/ContextPanel/components/FilesTab/useSessionFiles.ts`]:
    ["getMessageArtifacts(message)", "(query.data?.files ?? [])"],
  [`${COPILOT}/components/CopilotChatActionsProvider/CopilotChatActionsProvider.tsx`]:
    ["onBackendTurn: "],
  [`${COPILOT}/components/CopilotPendingReviews/CopilotPendingReviews.tsx`]: [
    "function handleReviewComplete(",
  ],
  [`${COPILOT}/components/SetupRequirementsCard/SetupRequirementsCard.tsx`]: [
    "extractInitialValues(",
    "buildSiblingInputsFromCredentials(",
  ],
  [`${COPILOT}/components/WorkspaceFileCards/useExpertDocuments.ts`]: [
    "(query.data?.files ?? [])",
  ],
  [`${COPILOT}/components/WorkspaceFileCards/useSessionActivity.ts`]: [
    "getSessionActivity(",
  ],
  [`${ORGANISMS}/Conversation/Conversation.tsx`]: [
    "function handleScrollToBottom(",
  ],
  [`${ORGANISMS}/PromptInput/PromptInput.tsx`]: [
    "function handleSubmit(",
    "function handleChange(",
    "function handleKeyDown(",
    "function handleClick(",
  ],
};

function compile(file: string) {
  const absolute = path.join(FRONTEND_ROOT, file);
  const result = babel.transformSync(readFileSync(absolute, "utf8"), {
    filename: absolute,
    babelrc: false,
    configFile: false,
    plugins: [
      syntaxJsx,
      [
        reactCompiler,
        {
          compilationMode: "infer",
          sources: reactCompilerSources,
          panicThreshold: "none",
        },
      ],
    ],
    presets: [[presetTypescript, { allowNamespaces: true }]],
  });
  return result?.code ?? "";
}

/** The cache check (`if ($[3] !== deps)`) of the block the line sits in, or
 *  null when it is computed on every render. */
function enclosingCacheCheck(lines: string[], index: number) {
  let depth = 0;
  for (let i = index - 1; i >= 0; i--) {
    depth += countOf(lines[i], "}") - countOf(lines[i], "{");
    if (depth >= 0) continue;
    if (/if \(\$\[\d+\]/.test(lines[i])) return lines[i].trim();
    if (/^\s*(export )?(function|const) \w+/.test(lines[i])) return null;
    depth = 0;
  }
  return null;
}

function countOf(line: string, char: string) {
  return line.split(char).length - 1;
}

describe("React Compiler caching in the copilot", () => {
  it.each(Object.entries(CACHED))("%s", (file, snippets) => {
    const lines = compile(file).split("\n");
    const uncached = snippets.filter((snippet) => {
      // Skips a module-level helper's own definition to find where it is
      // called.
      const index = lines.findIndex(
        (line) => line.includes(snippet) && !/^(export )?function /.test(line),
      );
      expect(index, `"${snippet}" is not in the compiled ${file}`).not.toBe(-1);
      return enclosingCacheCheck(lines, index) === null;
    });

    expect(uncached).toEqual([]);
  });
});
