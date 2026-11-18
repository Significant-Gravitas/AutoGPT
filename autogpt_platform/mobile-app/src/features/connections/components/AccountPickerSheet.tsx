import { Pressable, Text, View } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

import { BottomSheet } from "./BottomSheet";
import { ProviderTile } from "./ProviderTile";

export interface AccountChoice {
  rowKey: string;
  providerName: string;
  logoID: string | null;
  /** "allow" lets an expert use an account; "choose" picks one of several. */
  mode: "allow" | "choose";
  accounts: { id: string; title: string }[];
}

interface Props {
  choice: AccountChoice | null;
  onPick: (rowKey: string, account: { id: string; title: string }) => void;
  onClose: () => void;
}

/** Accounts the user already has for a service, to pick one. */
export function AccountPickerSheet({ choice, onPick, onClose }: Props) {
  const styles = useStyles();
  return (
    <BottomSheet isOpen={choice !== null} onClose={onClose}>
      {choice ? (
        <View style={styles.body}>
          <Text accessibilityRole="header" style={styles.title}>
            {choice.mode === "allow" ? "Allow an account" : "Choose an account"}
          </Text>
          <Text style={styles.intro}>
            {choice.mode === "allow"
              ? `Pick the ${choice.providerName} account to use here.`
              : `You have more than one ${choice.providerName} account. Pick the one to use in this chat.`}
          </Text>
          <View style={styles.list}>
            {choice.accounts.map((account, index) => (
              <Pressable
                key={account.id}
                accessibilityRole="button"
                onPress={() => onPick(choice.rowKey, account)}
                style={({ pressed }) => [
                  styles.row,
                  index > 0 && styles.divider,
                  pressed && styles.pressed,
                ]}
              >
                <ProviderTile name={choice.providerName} logoID={choice.logoID} />
                <Text style={styles.account} numberOfLines={1}>
                  {account.title}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
      ) : null}
    </BottomSheet>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  title: { ...type.heading, color: colors.text },
  intro: { ...type.small, fontSize: 14, lineHeight: 20, color: colors.textSecondary },
  list: {
    overflow: "hidden",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.md,
  },
  divider: { borderTopWidth: 1, borderTopColor: colors.border },
  pressed: { backgroundColor: colors.surfacePressed },
  account: { ...type.label, color: colors.text, flex: 1 },
}));
