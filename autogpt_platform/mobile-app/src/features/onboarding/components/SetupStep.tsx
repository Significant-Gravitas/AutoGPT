import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Image } from "expo-image";
import { ActivityIndicator, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { Toast } from "@/components/Toast";
import { OTTO_AVATAR_URL } from "@/lib/otto";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { checklistFor } from "../finishOnboarding";
import type { WorkAnswer } from "../helpers";
import { useSetupStep } from "../useSetupStep";

const STAGE_INDEX = { saving: 0, team: 1, finishing: 2 } as const;
const ITEM_COUNT = 3;

interface Props {
  userID: string;
  userName: string;
  role: string;
  work: WorkAnswer;
  recordingID: string;
  onPlanNeeded: () => void;
}

export function SetupStep(props: Props) {
  const setup = useSetupStep(props);
  const styles = useStyles();
  const colors = useColors();
  const items = checklistFor(props.work, setup.teamNames);
  const current = setup.isDone ? ITEM_COUNT : STAGE_INDEX[setup.stage];
  const progress = Math.min(1, (current + 1) / ITEM_COUNT);

  function renderMark(index: number) {
    if (index < current) {
      return (
        <View style={[styles.mark, styles.markDone]}>
          <Icon icon={Tick02Icon} size={12} color={colors.onBrand} />
        </View>
      );
    }
    if (index === current) {
      return (
        <View style={styles.markSlot}>
          <ActivityIndicator size="small" color={colors.text} />
        </View>
      );
    }
    return <View style={styles.mark} />;
  }

  return (
    <View style={styles.screen}>
      <Image
        source={OTTO_AVATAR_URL}
        style={styles.otto}
        contentFit="contain"
        accessibilityLabel="Otto"
      />
      <Text accessibilityRole="header" style={styles.heading}>
        Preparing your workspace...
      </Text>
      <View
        accessibilityRole="progressbar"
        accessibilityValue={{ min: 0, max: 100, now: Math.round(progress * 100) }}
        style={styles.track}
      >
        <View style={[styles.fill, { width: `${progress * 100}%` }]} />
      </View>
      {setup.error ? (
        <View style={styles.errorBox}>
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {setup.error}
          </Text>
          <Button label="Try again" onPress={setup.retry} />
        </View>
      ) : (
        <View style={styles.list}>
          {items.map((item, index) => (
            <View key={index} style={styles.item}>
              {renderMark(index)}
              <Text
                style={[
                  styles.itemText,
                  index < current && styles.itemDone,
                  index === current && styles.itemActive,
                ]}
              >
                {item}
              </Text>
            </View>
          ))}
        </View>
      )}
      {setup.toast ? <Toast toast={setup.toast} /> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xl,
    paddingHorizontal: spacing.xxl,
    backgroundColor: colors.background,
  },
  otto: { width: 104, height: 104 },
  heading: { ...type.headline, color: colors.text, textAlign: "center" },
  track: {
    alignSelf: "stretch",
    height: 2,
    overflow: "hidden",
    backgroundColor: colors.surfacePressed,
  },
  fill: { height: "100%", backgroundColor: colors.text },
  list: { gap: spacing.md },
  item: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  mark: {
    width: 20,
    height: 20,
    borderRadius: radius.pill,
    borderWidth: 1.5,
    borderColor: colors.borderStrong,
    alignItems: "center",
    justifyContent: "center",
  },
  markSlot: { width: 20, height: 20, alignItems: "center", justifyContent: "center" },
  markDone: { backgroundColor: colors.success, borderColor: colors.success },
  itemText: { ...type.body, color: colors.textTertiary },
  itemDone: { color: colors.text },
  itemActive: { color: colors.text, fontWeight: "500" },
  errorBox: { alignItems: "center", gap: spacing.lg },
  error: { ...type.body, color: colors.text, textAlign: "center" },
}));
