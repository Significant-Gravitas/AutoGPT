/**
 * Tutorial state management helpers
 */

import { Key, storage } from "@/services/storage/local-storage";
import { closeBlockMenu } from "./menu";
import { closeSaveControl, forceSaveOpen } from "./save";
import { removeAllHighlights, enableAllBlocks } from "./highlights";

/**
 * Handles tutorial cancellation
 */
export const handleTutorialCancel = (tour: any) => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  tour.cancel();
  storage.set(Key.SHEPHERD_TOUR, "canceled");
};

/**
 * Handles tutorial skip
 */
export const handleTutorialSkip = (tour: any) => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  tour.cancel();
  storage.set(Key.SHEPHERD_TOUR, "skipped");
};

/**
 * Handles tutorial completion
 */
export const handleTutorialComplete = () => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "completed");
};

