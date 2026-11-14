import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useAudioRecorderState, type AudioRecorder } from "expo-audio";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { elapsedLabel, meterLevel } from "../dictationHelpers";

// Fixed shapes scaled by the live level, so the bars move with the voice.
const BAR_WEIGHTS = [0.45, 0.8, 1, 0.65, 0.9, 0.5, 0.7];
const BAR_MIN = 4;
const BAR_RANGE = 20;

interface Props {
  recorder: AudioRecorder;
  isTranscribing: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function RecordingBar({ recorder, isTranscribing, onCancel, onConfirm }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const state = useAudioRecorderState(recorder, 100);
  const level = state.isRecording ? meterLevel(state.metering) : 0;

  return (
    <View style={styles.bar}>
      <IconButton
        icon={Cancel01Icon}
        label="Cancel dictation"
        onPress={onCancel}
        disabled={isTranscribing}
        variant="icon"
      />
      <View style={styles.meter} accessibilityElementsHidden importantForAccessibility="no">
        {BAR_WEIGHTS.map((weight, index) => (
          <View
            key={index}
            style={[styles.level, { height: BAR_MIN + BAR_RANGE * level * weight }]}
          />
        ))}
      </View>
      <Text style={styles.time}>{elapsedLabel(state.durationMillis)}</Text>
      <View style={styles.spacer} />
      {isTranscribing ? (
        <View style={styles.busy}>
          <ActivityIndicator color={colors.textSecondary} />
        </View>
      ) : (
        <IconButton
          icon={Tick02Icon}
          label="Use dictation"
          onPress={onConfirm}
          variant="primary"
        />
      )}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  bar: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.xs,
    paddingVertical: spacing.xs,
    borderRadius: radius.lg + 8,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  meter: { flexDirection: "row", alignItems: "center", gap: 3, height: BAR_MIN + BAR_RANGE },
  level: { width: 3, borderRadius: 2, backgroundColor: colors.brand },
  time: { ...type.small, color: colors.textSecondary, fontVariant: ["tabular-nums"] },
  spacer: { flex: 1 },
  busy: { width: 40, height: 40, alignItems: "center", justifyContent: "center" },
}));
