import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/views/chat/agent_message_tile.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_input_field.dart';
import 'package:auto_gpt_flutter_client/views/chat/user_message_tile.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';

class ChatView extends StatefulWidget {
  final ChatViewModel viewModel;

  const ChatView({Key? key, required this.viewModel}) : super(key: key);

  @override
  _ChatViewState createState() => _ChatViewState();
}

class _ChatViewState extends State<ChatView> {
  @override
  void initState() {
    super.initState();

    // Schedule the fetchTasks call for after the initial build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // TODO: Update to actual task id
      widget.viewModel.fetchChatsForTask(1);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Chat messages list
          Expanded(
            child: ListView.builder(
              itemCount: widget.viewModel.chats.length,
              itemBuilder: (context, index) {
                final chat = widget.viewModel.chats[index];
                if (chat.messageType == MessageType.user) {
                  return UserMessageTile(message: chat.message);
                } else {
                  return AgentMessageTile(message: chat.message);
                }
              },
            ),
          ),
          // Input area
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ChatInputField(
              onSendPressed: () {
                // TODO: Implement passing the message back up
              },
            ),
          ),
        ],
      ),
    );
  }
}
