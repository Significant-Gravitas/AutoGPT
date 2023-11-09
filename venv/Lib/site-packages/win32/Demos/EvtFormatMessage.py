import sys

import win32evtlog


def main():
    path = "System"
    num_events = 5
    if len(sys.argv) > 2:
        path = sys.argv[1]
        num_events = int(sys.argv[2])
    elif len(sys.argv) > 1:
        path = sys.argv[1]

    query = win32evtlog.EvtQuery(path, win32evtlog.EvtQueryForwardDirection)
    events = win32evtlog.EvtNext(query, num_events)
    context = win32evtlog.EvtCreateRenderContext(win32evtlog.EvtRenderContextSystem)

    for i, event in enumerate(events, 1):
        result = win32evtlog.EvtRender(
            event, win32evtlog.EvtRenderEventValues, Context=context
        )

        print("Event {}".format(i))

        level_value, level_variant = result[win32evtlog.EvtSystemLevel]
        if level_variant != win32evtlog.EvtVarTypeNull:
            if level_value == 1:
                print("    Level: CRITICAL")
            elif level_value == 2:
                print("    Level: ERROR")
            elif level_value == 3:
                print("    Level: WARNING")
            elif level_value == 4:
                print("    Level: INFO")
            elif level_value == 5:
                print("    Level: VERBOSE")
            else:
                print("    Level: UNKNOWN")

        time_created_value, time_created_variant = result[
            win32evtlog.EvtSystemTimeCreated
        ]
        if time_created_variant != win32evtlog.EvtVarTypeNull:
            print("    Timestamp: {}".format(time_created_value.isoformat()))

        computer_value, computer_variant = result[win32evtlog.EvtSystemComputer]
        if computer_variant != win32evtlog.EvtVarTypeNull:
            print("    FQDN: {}".format(computer_value))

        provider_name_value, provider_name_variant = result[
            win32evtlog.EvtSystemProviderName
        ]
        if provider_name_variant != win32evtlog.EvtVarTypeNull:
            print("    Provider: {}".format(provider_name_value))

            try:
                metadata = win32evtlog.EvtOpenPublisherMetadata(provider_name_value)
            # pywintypes.error: (2, 'EvtOpenPublisherMetadata', 'The system cannot find the file specified.')
            except Exception:
                pass
            else:
                try:
                    message = win32evtlog.EvtFormatMessage(
                        metadata, event, win32evtlog.EvtFormatMessageEvent
                    )
                # pywintypes.error: (15027, 'EvtFormatMessage: allocated 0, need buffer of size 0', 'The message resource is present but the message was not found in the message table.')
                except Exception:
                    pass
                else:
                    try:
                        print("    Message: {}".format(message))
                    except UnicodeEncodeError:
                        # Obscure error when run under subprocess.Popen(), presumably due to
                        # not knowing the correct encoding for the console.
                        # > UnicodeEncodeError: \'charmap\' codec can\'t encode character \'\\u200e\' in position 57: character maps to <undefined>\r\n'
                        # Can't reproduce when running manually, so it seems more a subprocess.Popen()
                        # than ours:
                        print(" Failed to decode:", repr(message))


if __name__ == "__main__":
    main()
