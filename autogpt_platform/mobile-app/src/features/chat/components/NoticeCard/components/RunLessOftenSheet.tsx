import { Tick02Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import type { ScheduledChatCadence } from "@/lib/api/scheduledChats";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { useRunLessOftenSheet } from "./useRunLessOftenSheet";

interface Props {
  scheduleKey: string;
  isOpen: boolean;
  onClose: () => void;
  /** Saved: the label of the cadence it runs on now. */
  onSaved: (label: string) => void;
}

export function RunLessOftenSheet({ scheduleKey, isOpen, onClose, onSaved }: Props) {
  return (
    <BottomSheet isOpen={isOpen} onClose={onClose}>
      {/* Mounted only while open, so each opening loads afresh. */}
      {isOpen ? <SheetBody scheduleKey={scheduleKey} onSaved={onSaved} /> : null}
    </BottomSheet>
  );
}

interface BodyProps {
  scheduleKey: string;
  onSaved: (label: string) => void;
}

function SheetBody({ scheduleKey, onSaved }: BodyProps) {
  const sheet = useRunLessOftenSheet({ scheduleKey, onSaved });
  const styles = useStyles();
  const colors = useColors();
  const { cadence } = sheet;

  function renderContent() {
    if (sheet.isLoading) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    if (sheet.loadError || !cadence) {
      return (
        <View style={styles.notice}>
          <Text style={styles.text}>{sheet.loadError}</Text>
          {sheet.canRetry ? (
            <Button label="Try again" onPress={sheet.retry} variant="secondary" size="small" />
          ) : null}
        </View>
      );
    }
    if (cadence.options.length === 0) {
      return <Text style={styles.text}>It already runs once a week or less.</Text>;
    }
    return (
      <>
        <Text style={styles.text}>{introText(cadence)}</Text>
        <View accessibilityRole="radiogroup" style={styles.options}>
          {cadence.options.map((option, index) => {
            const isSelected = option.cron === sheet.selected;
            return (
              <Pressable
                key={option.cron}
                accessibilityRole="radio"
                accessibilityLabel={option.label}
                accessibilityState={{ checked: isSelected }}
                onPress={() => sheet.choose(option.cron)}
                style={({ pressed }) => [
                  styles.option,
                  index > 0 && styles.divided,
                  pressed && styles.pressed,
                ]}
              >
                <Text style={styles.label}>{option.label}</Text>
                {isSelected ? <Icon icon={Tick02Icon} size={20} color={colors.text} /> : null}
              </Pressable>
            );
          })}
        </View>
        {sheet.saveError ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {sheet.saveError}
          </Text>
        ) : null}
        <Button
          label="Save"
          onPress={sheet.submit}
          isLoading={sheet.isSaving}
          disabled={!sheet.selected}
          fullWidth
        />
      </>
    );
  }

  return (
    <View style={styles.body}>
      <Text accessibilityRole="header" style={styles.heading}>
        {cadence ? `Run “${cadence.title}” less often` : "Run less often"}
      </Text>
      {renderContent()}
    </View>
  );
}

function introText(cadence: ScheduledChatCadence) {
  const now = cadence.current
    ? `It runs ${cadence.current.charAt(0).toLowerCase()}${cadence.current.slice(1)} now. `
    : "";
  return `${now}Fewer runs use less of your allowance. Times are in ${cadence.timezone}.`;
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  heading: { ...type.heading, color: colors.text },
  text: { ...type.small, color: colors.textSecondary },
  loading: { paddingVertical: spacing.xl },
  notice: { alignItems: "flex-start", gap: spacing.sm },
  options: {
    overflow: "hidden",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  option: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  divided: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  pressed: { backgroundColor: colors.surfacePressed },
  label: { ...type.body, flex: 1, color: colors.text },
  error: { ...type.small, color: colors.error },
}));
