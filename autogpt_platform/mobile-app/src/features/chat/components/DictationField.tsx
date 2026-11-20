import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { useAudioRecorderState, type AudioRecorder } from "expo-audio";
import { Pressable, Text, View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { VoiceOrb } from "@/components/VoiceOrb";
import { createStyles, spacing, type } from "@/lib/theme";

import { dictationLead, elapsedLabel, meterLevel } from "../dictationHelpers";
import { inputBox } from "./ComposerField";

interface Props {
  /** What is already typed; the dictated words go after it. */
  text: string;
  recorder: AudioRecorder;
  isTranscribing: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

/**
 * Stands in for the composer's field while dictating, at the same size: the
 * typed text stays in view and the mic button becomes the orb.
 */
export function DictationField({
  text,
  recorder,
  isTranscribing,
  onCancel,
  onConfirm,
}: Props) {
  const styles = useStyles();
  const state = useAudioRecorderState(recorder, 100);
  const level = state.isRecording ? meterLevel(state.metering) : 0;

  return (
    <View>
      {/* A long draft is cut at the top, keeping the end where the words land. */}
      <View style={styles.window}>
        <Text style={styles.text}>
          {dictationLead(text)}
          <Text style={styles.hint}>
            {isTranscribing ? "Transcribing…" : "Listening…"}
          </Text>
        </Text>
      </View>
      <View style={styles.row}>
        <IconButton
          icon={Cancel01Icon}
          label="Cancel dictation"
          onPress={onCancel}
          disabled={isTranscribing}
        />
        <Text style={styles.time}>{elapsedLabel(state.durationMillis)}</Text>
        <View style={styles.spacer} />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Use dictation"
          accessibilityState={{ disabled: isTranscribing, busy: isTranscribing }}
          disabled={isTranscribing}
          hitSlop={8}
          onPress={onConfirm}
          style={({ pressed }) => pressed && styles.pressed}
        >
          <VoiceOrb level={level} isBusy={isTranscribing} />
        </Pressable>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  window: {
    minHeight: inputBox.minHeight,
    maxHeight: inputBox.maxHeight,
    overflow: "hidden",
    justifyContent: "flex-end",
  },
  text: {
    ...type.body,
    ...inputBox,
    maxHeight: undefined,
    color: colors.text,
  },
  hint: { color: colors.textTertiary },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.sm,
  },
  time: {
    ...type.small,
    color: colors.textSecondary,
    fontVariant: ["tabular-nums"],
  },
  spacer: { flex: 1 },
  pressed: { opacity: 0.6 },
}));
