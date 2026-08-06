import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useState } from "react";

export function useInstallOnExpertButton() {
  const hireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const { isLoggedIn } = useAuth();
  const [pickerOpen, setPickerOpen] = useState(false);

  const canFetchExperts = Boolean(hireExpertsEnabled) && isLoggedIn;

  const expertsQuery = useListExperts({
    query: {
      select: (x) => x.data as Expert[],
      enabled: canFetchExperts,
    },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  return {
    canInstall: canFetchExperts && hiredExperts.length > 0,
    pickerOpen,
    openPicker: () => setPickerOpen(true),
    closePicker: () => setPickerOpen(false),
  };
}
