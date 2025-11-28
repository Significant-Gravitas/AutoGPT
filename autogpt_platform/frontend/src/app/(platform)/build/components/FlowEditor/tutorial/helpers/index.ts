/**
 * Tutorial helpers - re-exports all helper modules
 */

// DOM helpers
export {
  waitForElement,
  waitForInputValue,
  waitForSearchResult,
  waitForAnyBlockCard,
  focusElement,
  scrollIntoView,
  typeIntoInput,
  observeElement,
  watchSearchInput,
} from "./dom";

// Highlight helpers
export {
  disableOtherBlocks,
  enableAllBlocks,
  highlightElement,
  removeAllHighlights,
  pulseElement,
  highlightFirstBlockInSearch,
} from "./highlights";

// Block helpers
export {
  prefetchTutorialBlocks,
  getPrefetchedBlock,
  clearPrefetchedBlocks,
  addPrefetchedBlock,
  getNodeByBlockId,
  addAgentIOBlocks,
  getFormContainerSelector,
  getFormContainerElement,
} from "./blocks";

// Canvas helpers
export {
  waitForNodeOnCanvas,
  waitForNodesCount,
  getNodesCount,
  getFirstNode,
  getNodeById,
  nodeHasValues,
  fitViewToScreen,
} from "./canvas";

// Connection helpers
export { isConnectionMade } from "./connections";

// Menu helpers
export {
  forceBlockMenuOpen,
  openBlockMenu,
  closeBlockMenu,
  clearBlockMenuSearch,
} from "./menu";

// Save helpers
export {
  openSaveControl,
  closeSaveControl,
  forceSaveOpen,
  clickSaveButton,
  isAgentSaved,
} from "./save";

// State helpers
export {
  handleTutorialCancel,
  handleTutorialSkip,
  handleTutorialComplete,
} from "./state";

