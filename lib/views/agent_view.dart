import 'package:flutter/material.dart';

class AgentView extends StatelessWidget {
  const AgentView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.blue,
      child: const Center(
        child: Text(
          'Agents',
          style: TextStyle(fontSize: 24, color: Colors.white),
        ),
      ),
    );
  }
}
