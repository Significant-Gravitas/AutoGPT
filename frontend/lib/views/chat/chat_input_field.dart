import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/chat/continuous_mode_dialog.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ChatInputField extends StatefulWidget {
  // Callback to be triggered when the send button is pressed
  final Function(String) onSendPressed;
  final Function() onContinuousModePressed;
  final bool isContinuousMode;
  // TODO: Create a view model for this class and remove the ChatViewModel
  final ChatViewModel viewModel;

  const ChatInputField({
    Key? key,
    required this.onSendPressed,
    required this.onContinuousModePressed,
    this.isContinuousMode = false,
    required this.viewModel,
  }) : super(key: key);

  @override
  _ChatInputFieldState createState() => _ChatInputFieldState();
}

class _ChatInputFieldState extends State<ChatInputField> {
  // Controller for the TextField to manage its content
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();
  final FocusNode _throwawayFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _focusNode.addListener(() {
      if (_focusNode.hasFocus && widget.isContinuousMode) {
        widget.onContinuousModePressed();
      }
    });
  }

  @override
  void dispose() {
    _focusNode.dispose(); // Dispose of the FocusNode when you're done.
    super.dispose();
  }

  Future<void> _presentContinuousModeDialogIfNeeded() async {
    final showContinuousModeDialog = await widget.viewModel.prefsService
            .getBool('showContinuousModeDialog') ??
        true;

    FocusScope.of(context).requestFocus(_throwawayFocusNode);
    if (showContinuousModeDialog) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return ContinuousModeDialog(
            onProceed: () {
              Navigator.of(context).pop();
              _executeContinuousMode();
            },
            onCheckboxChanged: (bool value) async {
              await widget.viewModel.prefsService
                  .setBool('showContinuousModeDialog', !value);
            },
          );
        },
      );
    } else {
      _executeContinuousMode();
    }
  }

  void _executeContinuousMode() {
    if (!widget.isContinuousMode) {
      widget.onSendPressed(_controller.text);
      _controller.clear();
      _focusNode.unfocus();
    }
    widget.onContinuousModePressed();
  }

  @override
  Widget build(BuildContext context) {
    // Using LayoutBuilder to provide the current constraints of the widget,
    // ensuring it rebuilds when the window size changes
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate the width of the chat view based on the constraints provided
        double chatViewWidth = constraints.maxWidth;

        // Determine the width of the input field based on the chat view width.
        // If the chat view width is 1000 or more, the input width will be 900.
        // Otherwise, the input width will be the chat view width minus 40.
        double inputWidth = (chatViewWidth >= 1000) ? 900 : chatViewWidth - 40;

        return Container(
          width: inputWidth,
          // Defining the minimum and maximum height for the TextField container
          constraints: const BoxConstraints(
            minHeight: 50,
            maxHeight: 400,
          ),
          // Styling the container with a border and rounded corners
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border.all(color: Colors.black, width: 0.5),
            borderRadius: BorderRadius.circular(8),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8),
          // Using SingleChildScrollView to ensure the TextField can scroll
          // when the content exceeds its maximum height
          child: SingleChildScrollView(
            reverse: true,
            child: TextField(
              controller: _controller,
              focusNode: _focusNode,
              // Enable enter key stroke to send the message
              onSubmitted: (_) {
                widget.onSendPressed(_controller.text);
                _controller.clear();
              },
              // Allowing the TextField to expand vertically and accommodate multiple lines
              maxLines: null,
              decoration: InputDecoration(
                hintText: 'Type a message...',
                border: InputBorder.none,
                suffixIcon: Row(
                  mainAxisSize: MainAxisSize.min, // Set to minimum space
                  children: [
                    if (!widget.isContinuousMode)
                      Tooltip(
                        message: 'Send a single message',
                        child: IconButton(
                          splashRadius: 0.1,
                          icon: const Icon(Icons.send),
                          onPressed: () {
                            widget.onSendPressed(_controller.text);
                            _controller.clear();
                          },
                        ),
                      ),
                    Tooltip(
                      message: widget.isContinuousMode
                          ? ''
                          : 'Enable continuous mode',
                      child: IconButton(
                        splashRadius: 0.1,
                        icon: Icon(widget.isContinuousMode
                            ? Icons.pause
                            : Icons.fast_forward),
                        onPressed: () {
                          // TODO: All of this logic should be handled at a higher level in the widget tree. Temporary
                          if (!widget.isContinuousMode) {
                            _presentContinuousModeDialogIfNeeded();
                          } else {
                            widget.onContinuousModePressed();
                          }
                        },
                      ),
                    )
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
