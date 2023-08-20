import 'package:flutter/material.dart';

class ChatView extends StatelessWidget {
  const ChatView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.green,
      child: const Center(
        child: Text(
          'Chat',
          style: TextStyle(fontSize: 24, color: Colors.white),
        ),
      ),
    );
  }
}
