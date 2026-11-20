import { useAudioRecorderState, type AudioRecorder } from "expo-audio";
import { Text, View } from "react-native";

import { VoiceOrb } from "@/components/VoiceOrb";
import { elapsedLabel, meterLevel } from "@/features/chat/dictationHelpers";
import { createStyles, spacing, type } from "@/lib/theme";

import { OttoMightAsk } from "./OttoMightAsk";

interface Props {
  recorder: AudioRecorder;
  isTranscribing: boolean;
}

/** The body while talking: the time, the live orb and Otto's question. */
export function WorkListening({ recorder, isTranscribing }: Props) {
  const styles = useStyles();
  const state = useAudioRecorderState(recorder, 100);
  const level = state.isRecording ? meterLevel(state.metering) : 0;

  return (
    <>
      <Text style={styles.time}>{elapsedLabel(state.durationMillis)}</Text>
      <Text accessibilityLiveRegion="polite" style={styles.status}>
        {isTranscribing ? "Transcribing…" : "Listening…"}
      </Text>
      <View style={styles.orbArea}>
        <VoiceOrb size={112} level={level} isBusy={isTranscribing} />
      </View>
      <OttoMightAsk />
    </>
  );
}

const useStyles = createStyles((colors) => ({
  time: {
    ...type.display,
    color: colors.text,
    fontVariant: ["tabular-nums"],
  },
  status: { ...type.small, color: colors.textSecondary },
  orbArea: {
    flex: 1,
    minHeight: 150,
    alignSelf: "stretch",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: spacing.xl,
  },
}));
