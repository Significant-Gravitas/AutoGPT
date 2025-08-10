import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:flutter/material.dart';

class LoadingIndicator extends StatefulWidget {
  final bool isLoading;

  const LoadingIndicator({Key? key, required this.isLoading}) : super(key: key);

  @override
  _LoadingIndicatorState createState() => _LoadingIndicatorState();
}

class _LoadingIndicatorState extends State<LoadingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        double width =
            (constraints.maxWidth >= 1000) ? 850 : constraints.maxWidth - 65;

        return SizedBox(
          width: width,
          height: 4.0,
          child: widget.isLoading
              ? AnimatedBuilder(
                  animation: _controller,
                  builder: (context, child) {
                    return ShaderMask(
                      shaderCallback: (rect) {
                        return LinearGradient(
                          begin: Alignment.centerLeft,
                          end: Alignment.centerRight,
                          colors: [
                            Colors.grey[400]!,
                            AppColors.primaryLight,
                            Colors.white,
                            Colors.grey[400]!,
                          ],
                          stops: [
                            _controller.value - 0.5,
                            _controller.value - 0.25,
                            _controller.value,
                            _controller.value + 0.25,
                          ],
                        ).createShader(rect);
                      },
                      child: Container(
                        width: width,
                        height: 4.0,
                        color: Colors.white,
                      ),
                    );
                  },
                )
              : Container(
                  color: Colors.grey[400],
                ),
        );
      },
    );
  }
}
