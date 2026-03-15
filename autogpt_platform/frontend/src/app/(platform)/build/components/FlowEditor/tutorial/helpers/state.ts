import { Key, storage } from "@/services/storage/local-storage";
import { closeBlockMenu } from "./menu";
import { closeSaveControl, forceSaveOpen } from "./save";
import { removeAllHighlights, enableAllBlocks } from "./highlights";

const clearTutorialIntervals = () => {
  const intervalKeys = [
    "__tutorialCalcInterval",
    "__tutorialCheckInterval",
    "__tutorialSecondCalcInterval",
  ];

  intervalKeys.forEach((key) => {
    if ((window as any)[key]) {
      clearInterval((window as any)[key]);
      delete (window as any)[key];
    }
  });
};

export const handleTutorialCancel = (_tour?: any) => {
  clearTutorialIntervals();
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "canceled");
};

export const handleTutorialSkip = (_tour?: any) => {
  clearTutorialIntervals();
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "skipped");
};

export const handleTutorialComplete = () => {
  clearTutorialIntervals();
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "completed");
};
