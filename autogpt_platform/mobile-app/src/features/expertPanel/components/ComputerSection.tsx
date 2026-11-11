import { ComputerIcon } from "@hugeicons/core-free-icons";
import { View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, useColors } from "@/lib/theme";

import { ActivityRow } from "./ActivityRow";
import { ActivitySection } from "./ActivitySection";

interface Props {
  isLoading: boolean;
  expertName: string;
  state: string | null;
}

export function ComputerSection({ isLoading, expertName, state }: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <ActivitySection title="Computer" isLoading={isLoading}>
      <ActivityRow
        title={`${expertName}'s computer`}
        detail={state}
        trailing={
          <View style={styles.tray}>
            <Icon icon={ComputerIcon} size={20} color={colors.textTertiary} />
          </View>
        }
      />
    </ActivitySection>
  );
}

const useStyles = createStyles((colors) => ({
  tray: {
    width: 40,
    height: 32,
    borderRadius: radius.sm,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.surfaceRaised,
  },
}));
