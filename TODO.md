Create a class for handling input bindings more cleanly.
Write class handling ctypes mouse events.
Rewrite DataCollector to give specific names for files.
Use more stdin for getting training set names and model names.
retry data collection and bot on a larger sample size with tps camera.

Test TF and Keras model

Mess with pickle serialization of TrainingSample objects to remove stutter from writing in loop. https://docs.python.org/3/library/pickle.html
https://stackoverflow.com/questions/22002437/why-is-pickle-dumppickle-load-ipc-so-slow-and-are-there-fast-alternatives


Find a way to stabilize framerates or to make collected data framerate independent. (Get meaningful values, not just really small floats due to dt being small.)

Rework DataLoader to use KeybindHandler rather than dataEncoder for creating output samples.

Rework trainer to use ConvLSTM2D for image sequences rather than Conv layer for single images. (With the goal of removing areas where the bot 'forgets' what it was doing)

Figure out how generators work for datasets to avoid the gpu running out of memory during the evaluation step.

Figure out how to make the LSTM(s) take in variable sized sequences. Recall that the sequence length should be proportional to the framerate otherwise the model will not work correctly due to misjudging time-steps.

Add PlayVersusData.py to the cli args.

Add test suite for functions that we are uncertain of their in/outputs

Add easy to use gui or interface for distribution of just the data collector.

Move exit conditions to stateManager and have it be an event so that it can be used in threads.