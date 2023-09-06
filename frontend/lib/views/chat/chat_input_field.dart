import 'package:flutter/material.dart';

class ChatInputField extends StatefulWidget {
  // Callback to be triggered when the send button is pressed
  final Function(String) onSendPressed;

  const ChatInputField({
    Key? key,
    required this.onSendPressed,
  }) : super(key: key);

  @override
  _ChatInputFieldState createState() => _ChatInputFieldState();
}

class _ChatInputFieldState extends State<ChatInputField> {
  // Controller for the TextField to manage its content
  final TextEditingController _controller = TextEditingController();

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
              // Allowing the TextField to expand vertically and accommodate multiple lines
              maxLines: null,
              decoration: InputDecoration(
                hintText: 'Type a message...',
                border: InputBorder.none,
                suffixIcon: IconButton(
                  splashRadius: 0.1,
                  icon: const Icon(Icons.send),
                  onPressed: () {
                    // TODO: We allow empty messages?
                    if (_controller.text.isNotEmpty) {
                      widget.onSendPressed(_controller.text);
                      _controller.clear();
                    }
                  },
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
