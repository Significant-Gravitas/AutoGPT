import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { useState } from "react";
import { useSaveGraph } from "@/app/(platform)/build/hooks/useSaveGraph";

export const useScheduleGraph = () => {
  const { saveGraph } = useSaveGraph({
    showToast: false,
  });
  const hasInputs = useGraphStore(useShallow((state) => state.hasInputs));
  const hasCredentials = useGraphStore(
    useShallow((state) => state.hasCredentials),
  );
  const [openScheduleInputDialog, setOpenScheduleInputDialog] = useState(false);
  const [openCronSchedulerDialog, setOpenCronSchedulerDialog] = useState(false);

  const handleScheduleGraph = async () => {
    await saveGraph(undefined);
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
