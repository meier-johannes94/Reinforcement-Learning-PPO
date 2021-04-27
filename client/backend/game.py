import datetime
import os
from typing import List, Dict
import time

import numpy as np

class Game:

    def __init__(self,
                 *,
                 identifier : str,
                 player_one : str,
                 player_two : str,
                 fst_obs : List[float],
                 fst_action : List[float]
                ):

        self.identifier = identifier
        self.player_one = player_one
        self.player_two = player_two

        self.last_observation = fst_obs
        self.last_action = fst_action

        self.transition_buffer = []

    def add_transition(self, 
                       *,
                       next_obs : List[float],
                       next_action : List[float], 
                       r : int, 
                       done : int, 
                       info : Dict
                       ) -> None:
        
        self.transition_buffer.append([self.last_observation, 
                                       self.last_action, 
                                       next_obs, 
                                       r, 
                                       done, 
                                       info
                                      ]
                                     )

        self.last_observation = next_obs
        self.last_action = next_action

    def save(self,
             *,
             output_path : str
            ) -> None:

        now = datetime.datetime.now()

        path = os.path.join(output_path, 'games', str(now.year), str(now.month), str(now.day))
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, self.identifier), 
                 {'identifier': self.identifier,
                  'player_one': self.player_one,
                  'player_two': self.player_two,
                  'transitions': self.transition_buffer,
                  'timestamp': time.time()
                 }
                )
