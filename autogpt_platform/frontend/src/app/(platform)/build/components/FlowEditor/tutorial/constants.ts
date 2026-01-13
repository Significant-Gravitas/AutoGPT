// Block IDs for tutorial blocks
export const BLOCK_IDS = {
  CALCULATOR: "b1ab9b19-67a6-406d-abf5-2dba76d00c79",
  AGENT_INPUT: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  AGENT_OUTPUT: "363ae599-353e-4804-937e-b2ee3cef3da4",
} as const;

export const TUTORIAL_SELECTORS = {
  // Custom nodes - These are all before saving
  INPUT_NODE: '[data-id="custom-node-2"]',
  OUTPUT_NODE: '[data-id="custom-node-3 "]',
  CALCULATOR_NODE: '[data-id="custom-node-1"]',

  // Paricular field selector
  NAME_FIELD_OUTPUT_NODE: '[data-id="field-3-root_name"]',

  // Output Handlers
  SECOND_CALCULATOR_RESULT_OUTPUT_HANDLER:
    '[data-tutorial-id="output-handler-2-result"]',
  FIRST_CALCULATOR_RESULT_OUTPUT_HANDLER:
    '[data-tutorial-id="output-handler-1-result"]',

  // Input Handler
  SECOND_CALCULATOR_NUMBER_A_INPUT_HANDLER:
    '[data-tutorial-id="input-handler-2-a"]',
  OUTPUT_VALUE_INPUT_HANDLEER: '[data-tutorial-id="label-3-root_value"]',

  // Block Menu
  BLOCKS_TRIGGER: '[data-id="blocks-control-popover-trigger"]',
  BLOCKS_CONTENT: '[data-id="blocks-control-popover-content"]',
  BLOCKS_SEARCH_INPUT:
    '[data-id="blocks-control-search-bar"] input[type="text"]',
  BLOCKS_SEARCH_INPUT_BOX: '[data-id="blocks-control-search-bar"]',

  // Add a new selector that checks within search results

  // Block Menu Sidebar
  MENU_ITEM_INPUT_BLOCKS: '[data-id="menu-item-input_blocks"]',
  MENU_ITEM_ALL_BLOCKS: '[data-id="menu-item-all_blocks"]',
  MENU_ITEM_ACTION_BLOCKS: '[data-id="menu-item-action_blocks"]',
  MENU_ITEM_OUTPUT_BLOCKS: '[data-id="menu-item-output_blocks"]',
  MENU_ITEM_INTEGRATIONS: '[data-id="menu-item-integrations"]',
  MENU_ITEM_MY_AGENTS: '[data-id="menu-item-my_agents"]',
  MENU_ITEM_MARKETPLACE: '[data-id="menu-item-marketplace_agents"]',
  MENU_ITEM_SUGGESTION: '[data-id="menu-item-suggestion"]',

  // Block Cards
  BLOCK_CARD_PREFIX: '[data-id^="block-card-"]',
  BLOCK_CARD_AGENT_INPUT: '[data-id="block-card-AgentInputBlock"]',
  // Calculator block - legacy ID used in old tutorial
  BLOCK_CARD_CALCULATOR:
    '[data-id="block-card-b1ab9b1967a6406dabf52dba76d00c79"]',
  BLOCK_CARD_CALCULATOR_IN_SEARCH:
    '[data-id="blocks-control-search-results"] [data-id="block-card-b1ab9b1967a6406dabf52dba76d00c79"]',

  // Save Control
  SAVE_TRIGGER: '[data-id="save-control-popover-trigger"]',
  SAVE_CONTENT: '[data-id="save-control-popover-content"]',
  SAVE_AGENT_BUTTON: '[data-id="save-control-save-agent"]',
  SAVE_NAME_INPUT: '[data-id="save-control-name-input"]',
  SAVE_DESCRIPTION_INPUT: '[data-id="save-control-description-input"]',

  // Builder Actions (Run, Schedule, Outputs)
  BUILDER_ACTIONS: '[data-id="builder-actions"]',
  RUN_BUTTON: '[data-id="run-graph-button"]',
  STOP_BUTTON: '[data-id="stop-graph-button"]',
  SCHEDULE_BUTTON: '[data-id="schedule-graph-button"]',
  AGENT_OUTPUTS_BUTTON: '[data-id="agent-outputs-button"]',

  // Run Input Dialog
  RUN_INPUT_DIALOG_CONTENT: '[data-id="run-input-dialog-content"]',
  RUN_INPUT_CREDENTIALS_SECTION: '[data-id="run-input-credentials-section"]',
  RUN_INPUT_CREDENTIALS_FORM: '[data-id="run-input-credentials-form"]',
  RUN_INPUT_INPUTS_SECTION: '[data-id="run-input-inputs-section"]',
  RUN_INPUT_INPUTS_FORM: '[data-id="run-input-inputs-form"]',
  RUN_INPUT_ACTIONS_SECTION: '[data-id="run-input-actions-section"]',
  RUN_INPUT_MANUAL_RUN_BUTTON: '[data-id="run-input-manual-run-button"]',
  RUN_INPUT_SCHEDULE_BUTTON: '[data-id="run-input-schedule-button"]',

  // Custom Controls (bottom left)
  CUSTOM_CONTROLS: '[data-id="custom-controls"]',
  ZOOM_IN_BUTTON: '[data-id="zoom-in-button"]',
  ZOOM_OUT_BUTTON: '[data-id="zoom-out-button"]',
  FIT_VIEW_BUTTON: '[data-id="fit-view-button"]',
  LOCK_BUTTON: '[data-id="lock-button"]',
  TUTORIAL_BUTTON: '[data-id="tutorial-button"]',

  // Canvas
  REACT_FLOW_CANVAS: ".react-flow__pane",
  REACT_FLOW_NODE: ".react-flow__node",
  REACT_FLOW_NODE_FIRST: '[data-testid^="rf__node-"]:first-child',
  REACT_FLOW_EDGE: '[data-testid^="rf__edge-"]',

  // Node elements
  NODE_CONTAINER: '[data-id^="custom-node-"]',
  NODE_HEADER: '[data-id^="node-header-"]',
  NODE_INPUT_HANDLES: '[data-tutorial-id="input-handles"]',
  NODE_OUTPUT_HANDLE: '[data-handlepos="right"]',
  NODE_INPUT_HANDLE: "[data-nodeid]",
  FIRST_CALCULATOR_NODE_OUTPUT: '[data-tutorial-id="node-output"]',
  // These are the Id's of the nodes before saving
  CALCULATOR_NODE_FORM_CONTAINER: '[data-id^="form-creator-container-1-node"]', // <-- Add this line
  AGENT_INPUT_NODE_FORM_CONTAINER: '[data-id^="form-creator-container-2-node"]', // <-- Add this line
  AGENT_OUTPUT_NODE_FORM_CONTAINER:
    '[data-id^="form-creator-container-3-node"]', // <-- Add this line

  // Execution badges
  BADGE_QUEUED: '[data-id^="badge-"][data-id$="-QUEUED"]',
  BADGE_COMPLETED: '[data-id^="badge-"][data-id$="-COMPLETED"]',

  // Undo/Redo
  UNDO_BUTTON: '[data-id="undo-button"]',
  REDO_BUTTON: '[data-id="redo-button"]',
} as const;

export const CSS_CLASSES = {
  DISABLE: "new-builder-tutorial-disable",
  HIGHLIGHT: "new-builder-tutorial-highlight",
  PULSE: "new-builder-tutorial-pulse",
} as const;

export const TUTORIAL_CONFIG = {
  ELEMENT_CHECK_INTERVAL: 50, // ms
  INPUT_CHECK_INTERVAL: 100, // ms
  USE_MODAL_OVERLAY: true,
  SCROLL_BEHAVIOR: "smooth" as const,
  SCROLL_BLOCK: "center" as const,
  SEARCH_TERM_CALCULATOR: "Calculator",
} as const;
