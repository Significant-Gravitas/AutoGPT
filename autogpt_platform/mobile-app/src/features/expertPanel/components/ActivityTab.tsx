import { View } from "react-native";

import { createStyles, spacing } from "@/lib/theme";

import { useExpertActivity } from "../useExpertActivity";
import { ActivityEmptyState } from "./ActivityEmptyState";
import { ComputerSection } from "./ComputerSection";
import { RecentSection } from "./RecentSection";
import { UpcomingSection } from "./UpcomingSection";

interface Props {
  expertID: string;
  expertName: string;
  isOpen: boolean;
}

export function ActivityTab({ expertID, expertName, isOpen }: Props) {
  const { upcoming, recent, computer } = useExpertActivity(expertID, isOpen);
  const styles = useStyles();
  const sections = [upcoming, recent, computer];
  const isEmpty = sections.every((section) => !section.isLoading && !section.isShown);

  if (isEmpty) {
    return <ActivityEmptyState expertName={expertName} />;
  }

  return (
    <View style={styles.tab}>
      {upcoming.isLoading || upcoming.isShown ? (
        <UpcomingSection
          isLoading={upcoming.isLoading}
          schedules={upcoming.schedules}
          hiddenCount={upcoming.hiddenCount}
          onShowMore={upcoming.showMore}
          isPaused={upcoming.isPaused}
          onResume={upcoming.resume}
          isResuming={upcoming.isResuming}
          resumeError={upcoming.resumeError}
        />
      ) : null}
      {recent.isLoading || recent.isShown ? (
        <RecentSection isLoading={recent.isLoading} runs={recent.runs} />
      ) : null}
      {computer.isLoading || computer.isShown ? (
        <ComputerSection
          isLoading={computer.isLoading}
          expertName={expertName}
          state={computer.state}
        />
      ) : null}
    </View>
  );
}

const useStyles = createStyles(() => ({
  tab: { gap: spacing.xl },
}));
