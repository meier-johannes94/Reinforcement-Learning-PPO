import numpy as np

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

class RemoteBasicOpponent(BasicOpponent, RemoteControllerInterface):

    def __init__(self, weak, keep_mode=True):
        BasicOpponent.__init__(self, weak=weak, keep_mode=keep_mode)
        RemoteControllerInterface.__init__(self, identifier='StrongBasicOpponent')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs)
        

if __name__ == '__main__':
    controller = RemoteBasicOpponent(weak=False)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='user0', # Testuser
                    password='1234',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/user0', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=10
                   )

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queuing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
