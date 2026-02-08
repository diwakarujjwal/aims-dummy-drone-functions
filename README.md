# Dummy Drone Functions

This project makes use of Mediapipe and OpenCV to make dummy functions for a gesture controlled drone. It includes both a mediapipe only version and a mediapipe + ANN version.
Apart from basic features such as movement controls it features:
1) Utility Commands using left hand.
2) Backup movement control to right hand in absence of utility commands.
3) Utility functions like speed control and follow feature.
4) Safety implementation such as defaulting to HOVER when no hand is found on screen as well as proving confirmation time before command is executed.
