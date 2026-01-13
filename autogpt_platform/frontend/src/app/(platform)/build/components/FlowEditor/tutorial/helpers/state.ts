import { Key, storage } from "@/services/storage/local-storage";
import { closeBlockMenu } from "./menu";
import { closeSaveControl, forceSaveOpen } from "./save";
import { removeAllHighlights, enableAllBlocks } from "./highlights";

export const handleTutorialCancel = (_tour?: any) => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "canceled");
};

export const handleTutorialSkip = (_tour?: any) => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "skipped");
};

export const handleTutorialComplete = () => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "completed");
};
