import { Cancel01Icon } from "@hugeicons/core-free-icons";
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  TextInput,
  View,
} from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { AreaPicker } from "@/features/createExpert/components/AreaPicker";
import { CreateExpertFooter } from "@/features/createExpert/components/CreateExpertFooter";
import { JOB_TITLE_MAX_LENGTH, NAME_MAX_LENGTH } from "@/features/createExpert/helpers";
import { useCreateExpert } from "@/features/createExpert/useCreateExpert";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

const PREVIEW_SIZE = 84;

export default function CreateExpertScreen() {
  const form = useCreateExpert();
  const styles = useStyles();
  const colors = useColors();

  function handleCreate() {
    void form.create();
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.screen}
    >
      <ScreenHeader
        title="Create an expert"
        left={<IconButton icon={Cancel01Icon} label="Close" onPress={form.close} />}
      />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.preview}>
          <ExpertAvatar
            name={form.name || "Your expert"}
            source={form.previewSource}
            size={PREVIEW_SIZE}
            tint={form.previewTint}
          />
        </View>
        <View style={styles.fields}>
          <TextInput
            accessibilityLabel="Name"
            placeholder="Name your expert"
            placeholderTextColor={colors.textTertiary}
            selectionColor={colors.brand}
            value={form.name}
            onChangeText={form.setName}
            maxLength={NAME_MAX_LENGTH}
            autoCapitalize="words"
            autoCorrect={false}
            editable={!form.isCreating}
            style={styles.input}
          />
          <TextInput
            accessibilityLabel="What they look after"
            placeholder="What should they look after?"
            placeholderTextColor={colors.textTertiary}
            selectionColor={colors.brand}
            value={form.jobTitle}
            onChangeText={form.setJobTitle}
            maxLength={JOB_TITLE_MAX_LENGTH}
            editable={!form.isCreating}
            style={styles.input}
          />
        </View>
        <AreaPicker selected={form.family} onSelect={form.setFamily} />
      </ScrollView>
      <CreateExpertFooter
        error={form.error}
        canCreate={form.canCreate}
        isCreating={form.isCreating}
        onCancel={form.close}
        onCreate={handleCreate}
      />
    </KeyboardAvoidingView>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  scroll: { flex: 1 },
  content: {
    gap: spacing.xl,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.xl,
  },
  preview: {
    alignSelf: "center",
    padding: 2,
    borderRadius: PREVIEW_SIZE,
    borderWidth: 2,
    borderColor: colors.brand,
  },
  fields: { gap: spacing.sm },
  input: {
    ...type.body,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    color: colors.text,
  },
}));
