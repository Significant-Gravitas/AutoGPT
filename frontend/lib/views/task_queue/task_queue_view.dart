import 'package:flutter/material.dart';

class TestQueueView extends StatefulWidget {
  @override
  _TestQueueViewState createState() => _TestQueueViewState();
}

class _TestQueueViewState extends State<TestQueueView> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16.0),
      color: Colors.blueGrey[200], // Background color for visibility
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            'This is the TestQueueView',
            style: TextStyle(fontSize: 24),
          ),
          SizedBox(height: 16),
          Text(
            'Counter Value: $_counter',
            style: TextStyle(fontSize: 20),
          ),
          ElevatedButton(
            onPressed: _incrementCounter,
            child: Text('Increment Counter'),
          ),
        ],
      ),
    );
  }
}
