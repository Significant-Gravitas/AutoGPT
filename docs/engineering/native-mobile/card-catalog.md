# Native tool-card catalog

Status: verified source inventory; native implementation pending. Baseline `81d43f2982fe406acb7648f19c131e491c365d64` (2026-10-11). See [feature map](feature-map.md).

An AST scan of `backend/backend/copilot/tools/models.py` and annotated response classes under `backend/backend/copilot/tools/` finds **89 enum values, 88 values with model declarations, and 90 model declarations**. `mcp_guide` has no producer declaration. `agent_builder_guide` and `mcp_tool_output` each have two models. Runtime tool names and payload variants remain necessary; these counts alone do not prove all runtime outputs are typed.

## Presentation and action families

| Family                        | Required presentation/action                                                                                           | E2E scenario coverage                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Status and access             | Native status/error card; actionable sign-in or provider recovery only when server context authorizes it.              | Error/access/retry journey; optional fields and redaction.                                             |
| Workflow discovery and result | Native list, summary, run status and output details. Open in-app sheets, not Library.                                  | Open results; inspect run and outputs; deny cross-user target.                                         |
| Connection setup              | Native credential/provider form or consent sheet; complete connection then resume the originating action.              | Missing credentials, invalid key, OAuth cancel/return, reconnect, grant approval and stale action.     |
| Workflow construction in chat | Native proposal/step/validation summaries with supported save/run/confirm actions; no Build page.                      | Preview, clarification, confirmed save/run, validation failure and historical result.                  |
| Questions and intake          | Native single/multiple/custom answer controls; persisted pending/answered state.                                       | Answer, relaunch, replay and two-device stale-answer handling.                                         |
| Capabilities                  | Native capability rows/details and bounded outputs. Use native schema-backed input/setup when needed.                  | Discover, configure, execute, render structured/text/file outputs and errors.                          |
| Approvals                     | Native persisted review sheet with exact arguments, edits, references, spend, approve and decline.                     | Approve/decline/edit once; stale, expired, forbidden and concurrent actions.                           |
| Schedules and routines        | Native lists/status and focused configuration/action sheets accessible from chat.                                      | Create/configure, pause/resume/delete, timezone, reconnect and state reconciliation.                   |
| Expert resources              | Native expert resource/credential summaries with access-controlled grants and removal.                                 | Inspect, grant/revoke, preserve references and cross-expert ownership.                                 |
| MCP and delegation            | Dispatch by type plus tool/variant. Native discovery/result card or delegated-session status and navigation.           | MCP result vs sub-session variant, pending/completed/failure and unauthorized child chat.              |
| Documentation and web         | Native source/link/search/result cards; safe external navigation.                                                      | Long/rich text, source links, blocked URL schemes and missing results.                                 |
| Files and folders             | Native file/folder metadata and artifact viewer/share actions.                                                         | All artifact kinds, denied/deleted file, stale identifiers, paths and large content.                   |
| Browser and computer output   | Native action/status, screenshot artifact and terminal output; explicit isolated remote-content viewer when supported. | Screenshot retrieval, bounded terminal output, remote target validation and no credential bridge.      |
| Memory and planning           | Native memory summaries/forget confirmations and task progress.                                                        | Confirm destructive memory action, stale candidates, progress and raw tool-input todos.                |
| Channels and skills           | Native channel/skill rows and operation receipts with validated target navigation.                                     | Success/failure, long results, removed items and authorized links.                                     |
| AI company and sessions       | Native expert/confirmation/roster/consult/conversation cards. Company means experts.                                   | Hire/change confirmation, single-use token, expert details, delegation and saved conversation opening. |
| Historical reserved type      | No producing model found on this baseline. Preserve a bounded historical/unknown card state.                           | Historical transcript decodes without executing unknown actions.                                       |

A known type must have an explicit policy and scenario. Read-only receipts can use reusable native text/list/status primitives; a generic raw JSON dump does not substitute for an actionable card. Actions use persisted server state and ownership, never an arbitrary URL or command supplied in model output.

## Complete enum-to-model map

Paths below are relative to `autogpt_platform/`. Model references identify the inspected source declaration; line numbers identify the baseline and can drift.

| Wire `type`                          | Model declaration(s)                                                                                                                                                              | Native family                 |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `error`                              | `ErrorResponse` — `backend/backend/copilot/tools/models.py:375`                                                                                                                   | Status and access             |
| `no_results`                         | `NoResultsResponse` — `backend/backend/copilot/tools/models.py:245`                                                                                                               | Status and access             |
| `need_login`                         | `NeedLoginResponse` — `backend/backend/copilot/tools/models.py:368`                                                                                                               | Status and access             |
| `agents_found`                       | `AgentsFoundResponse` — `backend/backend/copilot/tools/models.py:235`                                                                                                             | Workflow discovery and result |
| `agent_details`                      | `AgentDetailsResponse` — `backend/backend/copilot/tools/models.py:296`                                                                                                            | Workflow discovery and result |
| `setup_requirements`                 | `SetupRequirementsResponse` — `backend/backend/copilot/tools/models.py:340`                                                                                                       | Connection setup              |
| `input_validation_error`             | `InputValidationErrorResponse` — `backend/backend/copilot/tools/models.py:539`                                                                                                    | Status and access             |
| `execution_started`                  | `ExecutionStartedResponse` — `backend/backend/copilot/tools/models.py:353`                                                                                                        | Workflow discovery and result |
| `agent_output`                       | `AgentOutputResponse` — `backend/backend/copilot/tools/models.py:567`                                                                                                             | Workflow discovery and result |
| `understanding_updated`              | `UnderstandingUpdatedResponse` — `backend/backend/copilot/tools/models.py:581`                                                                                                    | Status and access             |
| `suggested_goal`                     | `SuggestedGoalResponse` — `backend/backend/copilot/tools/models.py:872`                                                                                                           | Workflow construction in chat |
| `agent_builder_guide`                | `BuildingModeResponse` — `backend/backend/copilot/tools/enter_building_mode.py:24`; `AgentBuildingGuideResponse` — `backend/backend/copilot/tools/get_agent_building_guide.py:32` | Workflow construction in chat |
| `agent_builder_preview`              | `AgentPreviewResponse` — `backend/backend/copilot/tools/models.py:823`                                                                                                            | Workflow construction in chat |
| `agent_builder_saved`                | `AgentSavedResponse` — `backend/backend/copilot/tools/models.py:834`                                                                                                              | Workflow construction in chat |
| `agent_builder_clarification_needed` | `ClarificationNeededResponse` — `backend/backend/copilot/tools/models.py:846`                                                                                                     | Questions and intake          |
| `agent_builder_validation_result`    | `ValidationResultResponse` — `backend/backend/copilot/tools/models.py:1220`                                                                                                       | Workflow construction in chat |
| `agent_builder_fix_result`           | `FixResultResponse` — `backend/backend/copilot/tools/models.py:1229`                                                                                                              | Workflow construction in chat |
| `task_decomposition`                 | `TaskDecompositionResponse` — `backend/backend/copilot/tools/models.py:1349`                                                                                                      | Workflow construction in chat |
| `block_list`                         | `BlockListResponse` — `backend/backend/copilot/tools/models.py:961`                                                                                                               | Capabilities                  |
| `block_details`                      | `BlockDetailsResponse` — `backend/backend/copilot/tools/models.py:1008`                                                                                                           | Capabilities                  |
| `block_output`                       | `BlockOutputResponse` — `backend/backend/copilot/tools/models.py:1016`                                                                                                            | Capabilities                  |
| `review_required`                    | `ReviewRequiredResponse` — `backend/backend/copilot/tools/models.py:1030`                                                                                                         | Approvals                     |
| `approval_required`                  | `ApprovalRequiredResponse` — `backend/backend/copilot/tools/models.py:1042`                                                                                                       | Approvals                     |
| `capability_list`                    | `CapabilityListResponse` — `backend/backend/copilot/tools/models.py:974`                                                                                                          | Capabilities                  |
| `capability_details`                 | `CapabilityDetailsResponse` — `backend/backend/copilot/tools/models.py:987`                                                                                                       | Capabilities                  |
| `schedule_list`                      | `ScheduleListResponse` — `backend/backend/copilot/tools/manage_schedules.py:80`                                                                                                   | Schedules and routines        |
| `schedule_deleted`                   | `ScheduleDeletedResponse` — `backend/backend/copilot/tools/manage_schedules.py:119`                                                                                               | Schedules and routines        |
| `schedule_toggled`                   | `ScheduleToggledResponse` — `backend/backend/copilot/tools/manage_schedules.py:335`                                                                                               | Schedules and routines        |
| `expert_workflow`                    | `ExpertWorkflowResponse` — `backend/backend/copilot/tools/expert_resources.py:32`                                                                                                 | Expert resources              |
| `expert_workflows`                   | `ExpertWorkflowsResponse` — `backend/backend/copilot/tools/expert_resources.py:410`                                                                                               | Expert resources              |
| `expert_credentials`                 | `ExpertCredentialsResponse` — `backend/backend/copilot/tools/expert_resources.py:41`                                                                                              | Expert resources              |
| `routines`                           | `RoutinesResponse` — `backend/backend/copilot/tools/routines.py:59`                                                                                                               | Schedules and routines        |
| `routine`                            | `RoutineResponse` — `backend/backend/copilot/tools/routines.py:65`                                                                                                                | Schedules and routines        |
| `credential_grant_requested`         | `CredentialGrantRequestedResponse` — `backend/backend/copilot/tools/expert_resources.py:534`                                                                                      | Connection setup              |
| `schedule_created`                   | `ScheduleCreatedResponse` — `backend/backend/copilot/tools/schedule_followup.py:59`                                                                                               | Schedules and routines        |
| `agent_trigger_list`                 | `AgentTriggerListResponse` — `backend/backend/copilot/tools/list_agent_triggers.py:50`                                                                                            | Schedules and routines        |
| `trigger_setup`                      | `TriggerSetupResponse` — `backend/backend/copilot/tools/setup_agent_webhook_trigger.py:58`                                                                                        | Schedules and routines        |
| `trigger_config_required`            | `TriggerConfigRequiredResponse` — `backend/backend/copilot/tools/setup_agent_webhook_trigger.py:72`                                                                               | Connection setup              |
| `preset_list`                        | `PresetListResponse` — `backend/backend/copilot/tools/manage_presets.py:87`                                                                                                       | Schedules and routines        |
| `preset_updated`                     | `PresetUpdatedResponse` — `backend/backend/copilot/tools/manage_presets.py:97`                                                                                                    | Schedules and routines        |
| `preset_deleted`                     | `PresetDeletedResponse` — `backend/backend/copilot/tools/manage_presets.py:107`                                                                                                   | Schedules and routines        |
| `mcp_guide`                          | No producing model found                                                                                                                                                          | Historical reserved type      |
| `mcp_tools_discovered`               | `MCPToolsDiscoveredResponse` — `backend/backend/copilot/tools/models.py:1170`                                                                                                     | MCP and delegation            |
| `mcp_tool_output`                    | `SubSessionStatusResponse` — `backend/backend/copilot/tools/models.py:448`; `MCPToolOutputResponse` — `backend/backend/copilot/tools/models.py:1178`                              | MCP and delegation            |
| `doc_search_results`                 | `DocSearchResultsResponse` — `backend/backend/copilot/tools/models.py:900`                                                                                                        | Documentation and web         |
| `doc_page`                           | `DocPageResponse` — `backend/backend/copilot/tools/models.py:909`                                                                                                                 | Documentation and web         |
| `workspace_file_list`                | `WorkspaceFileListResponse` — `backend/backend/copilot/tools/workspace_files.py:188`                                                                                              | Files and folders             |
| `workspace_file_content`             | `WorkspaceFileContentResponse` — `backend/backend/copilot/tools/workspace_files.py:200`                                                                                           | Files and folders             |
| `workspace_file_metadata`            | `WorkspaceFileMetadataResponse` — `backend/backend/copilot/tools/workspace_files.py:298`                                                                                          | Files and folders             |
| `workspace_file_written`             | `WorkspaceWriteResponse` — `backend/backend/copilot/tools/workspace_files.py:311`                                                                                                 | Files and folders             |
| `workspace_file_deleted`             | `WorkspaceDeleteResponse` — `backend/backend/copilot/tools/workspace_files.py:327`                                                                                                | Files and folders             |
| `folder_created`                     | `FolderCreatedResponse` — `backend/backend/copilot/tools/models.py:1274`                                                                                                          | Files and folders             |
| `folder_list`                        | `FolderListResponse` — `backend/backend/copilot/tools/models.py:1281`                                                                                                             | Files and folders             |
| `folder_updated`                     | `FolderUpdatedResponse` — `backend/backend/copilot/tools/models.py:1291`                                                                                                          | Files and folders             |
| `folder_moved`                       | `FolderMovedResponse` — `backend/backend/copilot/tools/models.py:1298`                                                                                                            | Files and folders             |
| `folder_deleted`                     | `FolderDeletedResponse` — `backend/backend/copilot/tools/models.py:1306`                                                                                                          | Files and folders             |
| `agents_moved_to_folder`             | `AgentsMovedToFolderResponse` — `backend/backend/copilot/tools/models.py:1313`                                                                                                    | Files and folders             |
| `browser_navigate`                   | `BrowserNavigateResponse` — `backend/backend/copilot/tools/models.py:1191`                                                                                                        | Browser and computer output   |
| `browser_act`                        | `BrowserActResponse` — `backend/backend/copilot/tools/models.py:1200`                                                                                                             | Browser and computer output   |
| `browser_screenshot`                 | `BrowserScreenshotResponse` — `backend/backend/copilot/tools/models.py:1209`                                                                                                      | Browser and computer output   |
| `bash_exec`                          | `BashExecResponse` — `backend/backend/copilot/tools/models.py:1097`                                                                                                               | Browser and computer output   |
| `desktop_stream`                     | `DesktopStreamToolResponse` — `backend/backend/copilot/tools/models.py:1107`                                                                                                      | Browser and computer output   |
| `web_fetch`                          | `WebFetchResponse` — `backend/backend/copilot/tools/models.py:1055`                                                                                                               | Documentation and web         |
| `web_search`                         | `WebSearchResponse` — `backend/backend/copilot/tools/models.py:1080`                                                                                                              | Documentation and web         |
| `feature_request_search`             | `FeatureRequestSearchResponse` — `backend/backend/copilot/tools/models.py:1130`                                                                                                   | Documentation and web         |
| `feature_request_created`            | `FeatureRequestCreatedResponse` — `backend/backend/copilot/tools/models.py:1139`                                                                                                  | Documentation and web         |
| `memory_store`                       | `MemoryStoreResponse` — `backend/backend/copilot/tools/models.py:1368`                                                                                                            | Memory and planning           |
| `memory_search`                      | `MemorySearchResponse` — `backend/backend/copilot/tools/models.py:1375`                                                                                                           | Memory and planning           |
| `memory_forget_candidates`           | `MemoryForgetCandidatesResponse` — `backend/backend/copilot/tools/models.py:1383`                                                                                                 | Memory and planning           |
| `memory_forget_confirm`              | `MemoryForgetConfirmResponse` — `backend/backend/copilot/tools/models.py:1415`                                                                                                    | Memory and planning           |
| `todo_write`                         | `TodoWriteResponse` — `backend/backend/copilot/tools/models.py:1444`                                                                                                              | Memory and planning           |
| `platform_info`                      | `PlatformInfoResponse` — `backend/backend/copilot/tools/models.py:1457`                                                                                                           | Status and access             |
| `chat_platform_channel_list`         | `ChatPlatformChannelListResponse` — `backend/backend/copilot/tools/models.py:1478`                                                                                                | Channels and skills           |
| `chat_platform_posted`               | `ChatPlatformPostedResponse` — `backend/backend/copilot/tools/models.py:1487`                                                                                                     | Channels and skills           |
| `chat_platform_edited`               | `ChatPlatformEditedResponse` — `backend/backend/copilot/tools/models.py:1498`                                                                                                     | Channels and skills           |
| `skill_stored`                       | `StoreSkillResponse` — `backend/backend/copilot/tools/skills.py:2454`                                                                                                             | Channels and skills           |
| `skill_loaded`                       | `ReadSkillResponse` — `backend/backend/copilot/tools/skills.py:2474`                                                                                                              | Channels and skills           |
| `skill_deleted`                      | `DeleteSkillResponse` — `backend/backend/copilot/tools/skills.py:2489`                                                                                                            | Channels and skills           |
| `skill_list`                         | `ListSkillsResponse` — `backend/backend/copilot/tools/skills.py:2495`                                                                                                             | Channels and skills           |
| `expert_soul_updated`                | `ExpertSoulUpdatedResponse` — `backend/backend/copilot/tools/models.py:600`                                                                                                       | AI company and sessions       |
| `expert_change_proposed`             | `ExpertChangeProposedResponse` — `backend/backend/copilot/tools/models.py:787`                                                                                                    | AI company and sessions       |
| `expert_change_applied`              | `ExpertChangeAppliedResponse` — `backend/backend/copilot/tools/models.py:800`                                                                                                     | AI company and sessions       |
| `team_roster`                        | `TeamRosterResponse` — `backend/backend/copilot/tools/models.py:669`                                                                                                              | AI company and sessions       |
| `expert_chat_list`                   | `ExpertChatListResponse` — `backend/backend/copilot/tools/models.py:686`                                                                                                          | AI company and sessions       |
| `expert_chat_transcript`             | `ExpertChatTranscriptResponse` — `backend/backend/copilot/tools/models.py:708`                                                                                                    | AI company and sessions       |
| `expert_onboarding`                  | `ExpertOnboardingResponse` — `backend/backend/copilot/tools/models.py:863`                                                                                                        | Questions and intake          |
| `team_consult`                       | `ConsultVerdictResponse` — `backend/backend/copilot/tools/models.py:736`                                                                                                          | AI company and sessions       |
| `session_list`                       | `SessionListResponse` — `backend/backend/copilot/tools/models.py:766`                                                                                                             | AI company and sessions       |
| `session_message`                    | `SessionMessageResponse` — `backend/backend/copilot/tools/models.py:773`                                                                                                          | AI company and sessions       |

## Stream inventory

The stream enum in `backend/backend/copilot/response_model.py` contains **26 values**. Comments and terminal markers are wire behavior in addition to the JSON event discriminators.

| Enum                     | Wire value              |
| ------------------------ | ----------------------- |
| `START`                  | `start`                 |
| `FINISH`                 | `finish`                |
| `START_STEP`             | `start-step`            |
| `FINISH_STEP`            | `finish-step`           |
| `TEXT_START`             | `text-start`            |
| `TEXT_DELTA`             | `text-delta`            |
| `TEXT_END`               | `text-end`              |
| `REASONING_START`        | `reasoning-start`       |
| `REASONING_DELTA`        | `reasoning-delta`       |
| `REASONING_END`          | `reasoning-end`         |
| `TOOL_INPUT_START`       | `tool-input-start`      |
| `TOOL_INPUT_AVAILABLE`   | `tool-input-available`  |
| `TOOL_OUTPUT_AVAILABLE`  | `tool-output-available` |
| `TOOL_DISPLAY_AVAILABLE` | `data-tool-display`     |
| `ERROR`                  | `error`                 |
| `USAGE`                  | `usage`                 |
| `HEARTBEAT`              | `heartbeat`             |
| `STATUS`                 | `data-status`           |
| `CURSOR`                 | `data-cursor`           |
| `DREAM_OPERATIONS`       | `data-dream-operations` |
| `PENDING_DRAINED`        | `data-pending-drained`  |
| `FOLLOW_UP_QUEUED`       | `data-follow-up-queued` |
| `MODE_CHANGED`           | `data-mode-changed`     |
| `COMPACTION`             | `data-compaction`       |
| `PROVIDER_FAILURE`       | `data-provider-failure` |
| `CHECKPOINT`             | `data-checkpoint`       |

## Coverage outside the enum

- Historical tool names such as `find_block`, `run_block`, `continue_run_block`, `schedule_agent` and `run_mcp_tool`.
- SDK-native text results, `Read`/`Write` output and file diffs.
- Expert intake and expert-change parts rendered outside ordinary tool chains.
- Held-call result rows, expert-run metadata and context-compaction parts.
- Attached-file blocks, `workspace://` references and file-bearing tool results.
- Tool display titles, progress, structured provider failures and checkpoint/history reconciliation.
- Empty, malformed, oversized, unknown and forward-version payloads; render a bounded recoverable state without enabling unrecognized actions.

The existing 35-member `ToolResponseUnion` is only partial schema coverage. Most model `type` fields accept the broad enum with a default rather than a literal discriminator. Schema generation, runtime validation and native decoding need an explicit registry/variant design before treating these models as a public discriminated union.

No `render_ui`/OpenUI producer was found on this baseline. Additions from other branches enter the catalog and compatibility tests when they become part of the experiment.

## Regression gate

The implementation should mechanically compare current producer/stream declarations with the supported registry and representative persisted transcripts, require explicit native presentation and action policies for additions, and run the corresponding native UI and API E2E scenarios. This catalog records the inventory; it does not claim those tests or renderers already exist.
