# Highway_SimulationPlatform
## Carsim Simulation
If you have carsim, you can try to run client.py for a simulation on Carsim, including ego-car,vehicles flow,and roads.  
The part of interface for Carsim is not done by me,so I'm not sure whether it will work.
## Python Simulation
I'm in charge of this part.  
For this part, I use Pygame as basic simulation tool. You can just run highway_env/basic.py, and a simulation window will show up.  
On top of that, I also implement some simple Reinforcement Learning Algoritm for agent(our ego-car) with Tensorflow 1.x, like DQN and DDQN.  
Now I have implemented several main traffic circumstance, like highway, merge, roundabout and crossroad. The traffice flow is set to appear randomly, which will follw basic model——MOBIL.

#### This work is based on a project in NIPS workshop,provided by INRIA
