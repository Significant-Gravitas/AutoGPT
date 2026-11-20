import { Mic01Icon, PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, TextInput, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { MAX_TYPED_CHARS, canSendTyped, type WorkAnswer } from "../helpers";
import { useWorkStep } from "../useWorkStep";
import { OnboardingShell } from "./OnboardingShell";
import { OttoMightAsk } from "./OttoMightAsk";
import { WorkPrivacyNote } from "./WorkPrivacyNote";
import { WorkListening } from "./WorkListening";

const ORB_SIZE = 112;

interface Props {
  onSubmit: (answer: WorkAnswer) => void;
  onBackToRole: () => void;
}

export function WorkStep({ onSubmit, onBackToRole }: Props) {
  const step = useWorkStep({ onSubmit, onBackToRole });
  const styles = useStyles();
  const colors = useColors();
  const isListening = step.screen === "listening";

  function renderFooterButtons() {
    if (isListening) {
      return (
        <>
          <View style={styles.half}>
            <Button
              label="Start over"
              variant="secondary"
              onPress={step.startOver}
              disabled={step.isTranscribing}
              fullWidth
            />
          </View>
          <View style={styles.half}>
            <Button
              label="Send"
              onPress={step.sendSpoken}
              disabled={step.isTranscribing}
              fullWidth
            />
          </View>
        </>
      );
    }
    if (step.screen === "typing") {
      return (
        <>
          <View style={styles.half}>
            <Button
              label="Talk instead"
              variant="secondary"
              leadingIcon={Mic01Icon}
              onPress={step.talk}
              fullWidth
            />
          </View>
          <View style={styles.half}>
            <Button
              label="Send"
              onPress={step.sendTyped}
              disabled={!canSendTyped(step.typed)}
              fullWidth
            />
          </View>
        </>
      );
    }
    return (
      <>
        <View style={styles.half}>
          <Button
            label="Talk"
            leadingIcon={Mic01Icon}
            onPress={step.talk}
            fullWidth
          />
        </View>
        <View style={styles.half}>
          <Button
            label="Type instead"
            variant="secondary"
            leadingIcon={PencilEdit02Icon}
            onPress={step.write}
            fullWidth
          />
        </View>
      </>
    );
  }

  function renderBody() {
    if (isListening) {
      return (
        <WorkListening
          recorder={step.recorder}
          isTranscribing={step.isTranscribing}
        />
      );
    }
    if (step.screen === "typing") {
      return (
        <>
          <Text accessibilityRole="header" style={styles.heading}>
            Write to me about your work
          </Text>
          <Text style={[styles.copy, styles.typingCopy]}>
            What repeats every week, and what would you hand off first?{" "}
            <Text style={styles.strong}>A few sentences is plenty.</Text>
          </Text>
          {step.notice ? (
            <Text accessibilityLiveRegion="polite" style={styles.notice}>
              {step.notice}
            </Text>
          ) : null}
          <TextInput
            accessibilityLabel="Your work"
            placeholder="I run a small studio. Every Monday I…"
            placeholderTextColor={colors.textTertiary}
            selectionColor={colors.brand}
            value={step.typed}
            onChangeText={step.setTyped}
            maxLength={MAX_TYPED_CHARS}
            multiline
            textAlignVertical="top"
            style={styles.field}
          />
        </>
      );
    }
    return (
      <>
        <Text accessibilityRole="header" style={styles.heading}>
          Talk to me about your work
        </Text>
        <Text style={styles.copy}>
          Tell me what you do and what eats up your week.{" "}
          <Text style={styles.strong}>
            Otto uses it to pick your experts and plan your first week.
          </Text>{" "}
          A minute is plenty, and you can type instead.
        </Text>
        <View style={styles.orbArea}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Talk"
            onPress={step.talk}
            style={({ pressed }) => [styles.halo, pressed && styles.pressed]}
          >
            <View style={styles.orb}>
              <Icon icon={Mic01Icon} size={40} color={colors.onBrand} />
            </View>
          </Pressable>
        </View>
        <OttoMightAsk />
      </>
    );
  }

  return (
    <OnboardingShell
      filled={2}
      onBack={step.isTranscribing ? undefined : step.back}
      onSkip={isListening ? undefined : step.skip}
      footer={
        <>
          <View style={styles.row}>{renderFooterButtons()}</View>
          <WorkPrivacyNote />
        </>
      }
    >
      {renderBody()}
    </OnboardingShell>
  );
}

const HALO = 14;

const useStyles = createStyles((colors) => ({
  heading: { ...type.headline, color: colors.text, textAlign: "center" },
  copy: {
    ...type.body,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.sm,
  },
  typingCopy: { marginBottom: spacing.lg },
  strong: { color: colors.text, fontWeight: "500" },
  notice: {
    ...type.small,
    alignSelf: "stretch",
    color: colors.error,
    marginBottom: spacing.sm,
  },
  field: {
    ...type.body,
    alignSelf: "stretch",
    height: 168,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    color: colors.text,
  },
  orbArea: {
    flex: 1,
    minHeight: 150,
    alignSelf: "stretch",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: spacing.xl,
  },
  halo: {
    padding: HALO,
    borderRadius: radius.pill,
    backgroundColor: colors.brandSoft,
  },
  orb: {
    width: ORB_SIZE,
    height: ORB_SIZE,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.brand,
  },
  row: { flexDirection: "row", gap: spacing.md },
  half: { flex: 1 },
  pressed: { opacity: 0.85 },
}));
