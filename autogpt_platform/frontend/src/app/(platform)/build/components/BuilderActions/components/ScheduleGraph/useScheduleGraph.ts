import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { useNewSaveControl } from "../../../NewControlPanel/NewSaveControl/useNewSaveControl";
import { useState } from "react";

export const useScheduleGraph = () => {
  const { onSubmit: onSaveGraph } = useNewSaveControl({
    showToast: false,
  });
  const hasInputs = useGraphStore(useShallow((state) => state.hasInputs));
  const hasCredentials = useGraphStore(
    useShallow((state) => state.hasCredentials),
  );
  const [openScheduleInputDialog, setOpenScheduleInputDialog] = useState(false);
  const [openCronSchedulerDialog, setOpenCronSchedulerDialog] = useState(false);

  const handleScheduleGraph = async () => {
    await onSaveGraph(undefined);
    if (hasInputs() || hasCredentials()) {
      setOpenScheduleInputDialog(true);
    } else {
      setOpenCronSchedulerDialog(true);
    }
  };

  return {
    openScheduleInputDialog,
    setOpenScheduleInputDialog,
    handleScheduleGraph,
    openCronSchedulerDialog,
    setOpenCronSchedulerDialog,
  };
};
