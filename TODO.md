Create a class for handling input bindings more cleanly. [x]
Write class handling ctypes mouse events. [x]
Rewrite DataCollector to give specific names for files. [x]
Create CLI for this tool. [x]

Mess with pickle serialization of TrainingSample objects to remove stutter from writing in loop. https://docs.python.org/3/library/pickle.html
https://stackoverflow.com/questions/22002437/why-is-pickle-dumppickle-load-ipc-so-slow-and-are-there-fast-alternatives [x]


Find a way to stabilize framerates or to make collected data framerate independent. (Get meaningful values, not just really small floats due to dt being small.) -> Not applicable for now.

Rework DataLoader to use KeybindHandler rather than dataEncoder for creating output samples. [x]

Move exit conditions to stateManager and have it be an event so that it can be used in threads. [x]

Rework trainer to use ConvLSTM2D for image sequences rather than Conv layer for single images. (With the goal of removing areas where the bot 'forgets' what it was doing)

Figure out how generators work for datasets to avoid the gpu running out of memory during the evaluation step.

Figure out how to make the LSTM(s) take in variable sized sequences. Recall that the sequence length should be proportional to the framerate otherwise the model will not work correctly due to misjudging time-steps.

Add PlayVersusData.py to the cli args.

Add test suite for functions that we are uncertain of their in/outputs

Add easy to use gui or interface for distribution of just the data collector.

Look into DQfD learning which combines imitation learning and reinforcement learning.

Change output layer to use different activation and loss functions for the x,y outputs to avoid averaging mouse movement at 0,0


    Contact this prof:
    Farhad_Pourkamali@uml.edu

Currently the input to the model is (240, 135) but in the paper, the input is (180, 80) While this may impact the memory usage,
I doubt this will fix the memory issues we're encountering - especially if this person used 64 frame history per sample.