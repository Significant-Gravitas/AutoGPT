import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import type { CredentialMeta } from "@/lib/api/integrations";
import { createStyles, radius, spacing, type } from "@/lib/theme";

import { logoID } from "../../helpers";
import { BottomSheet } from "../BottomSheet";
import { ProviderTile } from "../ProviderTile";
import { SchemePicker } from "./components/SchemePicker";
import { SheetField } from "./components/SheetField";
import { sheetIntro, type SheetTarget } from "./helpers";
import { useCredentialSheet } from "./useCredentialSheet";

interface Props {
  /** The service to add a key, login, header or token for; null keeps it closed. */
  target: SheetTarget | null;
  onClose: () => void;
  onSaved: (rowKey: string, credential: CredentialMeta) => void;
}

/** One sheet for every credential typed in by hand. Fields start empty each time. */
export function CredentialSheet({ target, onClose, onSaved }: Props) {
  return (
    <BottomSheet isOpen={target !== null} onClose={onClose}>
      {target ? (
        <CredentialForm
          key={`${target.rowKey}:${target.kind}`}
          target={target}
          onSaved={onSaved}
        />
      ) : null}
    </BottomSheet>
  );
}

interface FormProps {
  target: SheetTarget;
  onSaved: (rowKey: string, credential: CredentialMeta) => void;
}

export function CredentialForm({ target, onSaved }: FormProps) {
  const styles = useStyles();
  const sheet = useCredentialSheet({ target, onSaved });
  const { values, change } = sheet;

  function renderFields() {
    switch (target.kind) {
      case "api_key":
        return (
          <>
            <SheetField
              label="API key"
              value={values.secret}
              onChangeText={change("secret")}
              isSecret
              autoFocus
            />
            <SheetField
              label="Name (optional)"
              value={values.title}
              onChangeText={change("title")}
              placeholder={`${target.providerName} key`}
              onSubmitEditing={sheet.submit}
            />
          </>
        );
      case "user_password":
        return (
          <>
            <SheetField
              label="Username"
              value={values.username}
              onChangeText={change("username")}
              autoFocus
            />
            <SheetField
              label="Password"
              value={values.secret}
              onChangeText={change("secret")}
              isSecret
            />
            {sheet.schemes ? (
              <>
                {sheet.schemes.length > 1 ? (
                  <SchemePicker
                    schemes={sheet.schemes}
                    value={values.scheme}
                    onChange={change("scheme")}
                  />
                ) : null}
                <SheetField
                  label="Server"
                  value={values.server}
                  onChangeText={change("server")}
                  placeholder="host or host:port"
                />
              </>
            ) : null}
            <SheetField
              label="Name (optional)"
              value={values.title}
              onChangeText={change("title")}
              placeholder={`My ${target.providerName} account`}
            />
          </>
        );
      case "host_scoped":
        return (
          <>
            <SheetField
              label="Host"
              value={values.host}
              onChangeText={change("host")}
              placeholder="api.example.com"
              autoFocus={!target.host}
            />
            <SheetField
              label="Header name"
              value={values.headerName}
              onChangeText={change("headerName")}
            />
            <SheetField
              label="Header value"
              value={values.secret}
              onChangeText={change("secret")}
              isSecret
              autoFocus={!!target.host}
            />
          </>
        );
      case "mcp_token":
        return (
          <>
            <View style={styles.server}>
              <Text style={styles.serverLabel}>Server</Text>
              <Text style={styles.serverURL} selectable>
                {target.serverURL}
              </Text>
            </View>
            <Text style={styles.intro}>
              Only paste a token if you know and trust this server. It will
              receive the token.
            </Text>
            <SheetField
              label="Token"
              value={values.secret}
              onChangeText={change("secret")}
              isSecret
              autoFocus
              onSubmitEditing={sheet.submit}
            />
          </>
        );
    }
  }

  return (
    <View style={styles.form}>
      <View style={styles.heading}>
        <ProviderTile name={target.providerName} logoID={logoID(target.provider, target.serverURL)} />
        <Text accessibilityRole="header" style={styles.title} numberOfLines={1}>
          {target.providerName}
        </Text>
      </View>
      <Text style={styles.intro}>{target.notice ?? sheetIntro(target.kind)}</Text>
      {renderFields()}
      {sheet.error ? <Text style={styles.error}>{sheet.error}</Text> : null}
      <Button
        label="Save"
        onPress={sheet.submit}
        disabled={!sheet.canSave}
        isLoading={sheet.isSaving}
        fullWidth
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  form: { gap: spacing.md },
  heading: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  title: { ...type.heading, color: colors.text, flex: 1 },
  intro: {
    ...type.small,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textSecondary,
  },
  error: { ...type.small, color: colors.error },
  server: {
    gap: spacing.xs,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  serverLabel: { ...type.small, color: colors.textTertiary },
  serverURL: { ...type.label, color: colors.text },
}));
