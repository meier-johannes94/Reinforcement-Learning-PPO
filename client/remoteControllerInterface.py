from abc import ABC, abstractmethod

import numpy as np

class RemoteControllerInterface(ABC):

    def __init__(self, 
                 identifier : str,
                ) -> None:

        """
        Plaese use the identifier to specify the Algorithm you are using 
        """
            
        self.identifier = identifier

    @abstractmethod
    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        """
        Expects an observation as input, returns an action
        """

        raise NotImplementedError()

    def before_game_starts(self) -> None:
        """
        Called before a new game.
        In case of a stateful policy, e.g. recurrent policy, this function can be used
        to reset the policy before a new game
        """

        pass

    def after_game_ends(self) -> None:
        """
        Called after every game
        """

        pass
        
        