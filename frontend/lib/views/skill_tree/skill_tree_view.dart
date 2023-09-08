import 'package:flutter/material.dart';

class SkillTreeView extends StatelessWidget {
  const SkillTreeView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.blue[100], // Background color
      child: const Center(
        child: Text(
          'SkillTreeView',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
