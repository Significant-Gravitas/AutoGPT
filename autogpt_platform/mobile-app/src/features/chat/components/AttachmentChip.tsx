import {
  AlertCircleIcon,
  Cancel01Icon,
  File01Icon,
  Image01Icon,
} from "@hugeicons/core-free-icons";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import type { Attachment } from "../attachmentHelpers";

interface Props {
  attachment: Attachment;
  onRemove?: (id: string) => void;
  onRetry?: (id: string) => void;
}

export function AttachmentChip({ attachment, onRemove, onRetry }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const { id, name, status, isImage } = attachment;
  const isFailed = status === "failed";

  function handleRetry() {
    onRetry?.(id);
  }

  function handleRemove() {
    onRemove?.(id);
  }

  return (
    <View style={styles.chip}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={isFailed ? `Try attaching ${name} again` : name}
        disabled={!isFailed || !onRetry}
        onPress={handleRetry}
        style={styles.body}
      >
        <Icon
          icon={isImage ? Image01Icon : File01Icon}
          size={16}
          color={colors.textSecondary}
        />
        <Text style={styles.name} numberOfLines={1} ellipsizeMode="middle">
          {name}
        </Text>
        {status === "uploading" ? (
          <ActivityIndicator size="small" color={colors.textSecondary} />
        ) : null}
        {isFailed ? <Icon icon={AlertCircleIcon} size={16} color={colors.error} /> : null}
      </Pressable>
      {onRemove ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Remove ${name}`}
          hitSlop={8}
          onPress={handleRemove}
          style={({ pressed }) => [styles.remove, pressed && styles.pressed]}
        >
          <Icon icon={Cancel01Icon} size={14} color={colors.textSecondary} />
        </Pressable>
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    height: 34,
    paddingLeft: spacing.md,
    paddingRight: spacing.sm,
    borderRadius: radius.sm + 2,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  body: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.small, color: colors.text, maxWidth: 180 },
  remove: { padding: 2 },
  pressed: { opacity: 0.6 },
}));
