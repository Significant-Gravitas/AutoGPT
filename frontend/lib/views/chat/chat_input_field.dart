import 'package:flutter/material.dart';

class ChatInputField extends StatefulWidget {
  // Callback to be triggered when the send button is pressed
  final Function(String) onSendPressed;
  final Function() onContinuousModePressed;
  final bool isContinuousMode;

  const ChatInputField({
    Key? key,
    required this.onSendPressed,
    required this.onContinuousModePressed,
    this.isContinuousMode = false,
  }) : super(key: key);

  @override
  _ChatInputFieldState createState() => _ChatInputFieldState();
}

class _ChatInputFieldState extends State<ChatInputField> {
  // Controller for the TextField to manage its content
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();

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
                    // TODO: Include pop up to explain continuous mode reprecussions
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
                          if (!widget.isContinuousMode) {
                            widget.onSendPressed(_controller.text);
                            _controller.clear();
                            _focusNode.unfocus();
                          }
                          widget.onContinuousModePressed();
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
