# DPPO-tensorflow
DPPO-tensorflow for Atari games based on baselines
-----------------------------------------------------------------------------------------------------------------
This project is a tensorflow version of DPPO for atari games. The reference is https://github.com/lnpalmer/PPO.
I rewrote lnpalmer's code from torch to tensorflow.


Denpendencies:
--------------
<br>Tensorflow >=1.7.0</br>
<br>python 3.5</br>
<br>baselines 0.1.5</br>

---------
Usage:
---------
<br>The default environemnt is PongNoFrameskip-v4.If you want to test its performs in Pong</br>
<br>In terminal,just typing:</br>
<br>`python runner.py`</br>
<br>Of course, you can type `python runner.py --env_id <env_name>` to change environments.</br> 
