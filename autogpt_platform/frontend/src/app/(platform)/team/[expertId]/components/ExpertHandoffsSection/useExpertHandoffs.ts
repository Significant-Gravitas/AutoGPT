import { useDelegationPages } from "../../../components/DelegationList/useDelegationPages";
import { getDelegationTotal } from "../../../components/DelegationList/helpers";

/** The hand-offs Otto (or a teammate) gave this expert, newest first. */
export function useExpertHandoffs(expertId: string) {
  const pages = useDelegationPages({ expertId });

  return {
    ...pages,
    total: pages.summary ? getDelegationTotal(pages.summary) : 0,
    isWorkingForOtto: pages.delegations.some(
      (d) => d.status === "running" && !d.delegated_by_expert_id,
    ),
  };
}
