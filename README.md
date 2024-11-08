## Explanation of the Simulation
Here you can see an example Environment.   The white dots are the agents, which observe and then act on their environment in each timestep.
Each one of those agents has their own small neural network for this task.
At the left and right edges you can see a grey area with is called the zone. When a fixed number of timesteps is finished, any agents that are currently not inside the zone will get removed.
The remaining ones will be the survivors which will pass on their genes. Their children will then start the next generation. And the cycle continues. 

![Screenshot 2024-11-06 194617](https://github.com/user-attachments/assets/25c947ae-5bb8-4079-99a4-2261b2d6d775)

With enough generations their neural networks will evolve,   
to make them more likely to manouver into the zone.  
Here is an example of how one of these brains can look like:  

![Dot Brain drawio(2)](https://github.com/user-attachments/assets/90a6045d-484e-4e44-8e2a-bc4f68656786)


