import { ActionSheetIOS, Alert, Platform, type AlertButton } from "react-native";

export interface ActionOption<T extends string> {
  id: T;
  label: string;
  isDestructive?: boolean;
}

// An Android alert holds three buttons at most, so a longer menu drops its
// Cancel button; a tap outside or Back still closes it.
const ANDROID_MAX_BUTTONS = 3;

/** A native menu of options; resolves with the chosen id, or null on cancel. */
export function chooseAction<T extends string>(
  title: string,
  options: ActionOption<T>[],
) {
  return new Promise<T | null>((resolve) => {
    if (Platform.OS === "ios") {
      ActionSheetIOS.showActionSheetWithOptions(
        {
          title,
          options: [...options.map((option) => option.label), "Cancel"],
          cancelButtonIndex: options.length,
          destructiveButtonIndex: options.flatMap((option, index) =>
            option.isDestructive ? [index] : [],
          ),
        },
        (index) => resolve(options[index]?.id ?? null),
      );
      return;
    }
    // Android lays buttons out right to left, so the first option sits last.
    const buttons: AlertButton[] = options
      .map<AlertButton>((option) => ({
        text: option.label,
        style: option.isDestructive ? "destructive" : "default",
        onPress: () => resolve(option.id),
      }))
      .reverse();
    if (buttons.length < ANDROID_MAX_BUTTONS) {
      buttons.unshift({ text: "Cancel", style: "cancel", onPress: () => resolve(null) });
    }
    Alert.alert(title, undefined, buttons, {
      cancelable: true,
      onDismiss: () => resolve(null),
    });
  });
}
