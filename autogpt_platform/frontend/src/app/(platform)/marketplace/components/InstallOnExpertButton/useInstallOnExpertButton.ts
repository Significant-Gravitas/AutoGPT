import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useState } from "react";

export function useInstallOnExpertButton() {
  const hireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const [pickerOpen, setPickerOpen] = useState(false);

  const expertsQuery = useListExperts({
    query: {
      select: (x) => x.data as Expert[],
      enabled: Boolean(hireExpertsEnabled),
    },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  return {
    canInstall: Boolean(hireExpertsEnabled) && hiredExperts.length > 0,
    pickerOpen,
    openPicker: () => setPickerOpen(true),
    closePicker: () => setPickerOpen(false),
  };
}
