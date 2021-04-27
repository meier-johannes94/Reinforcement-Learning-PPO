import numpy as np
import argparse
from typing import Dict, List, Optional

import laserhockey

from twisted.internet import reactor, task

from .network_interface import NetworkInterface, NetworkInterfaceConnectionError
from .client_cmd import ClientCMD, InteractiveMode
from .game import Game
from client.remoteControllerInterface import RemoteControllerInterface


def parseOptions():
    parser = argparse.ArgumentParser(description='ALRL2021 Competition Client.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', '--username', action='store',
                         type=str, dest='username', default="",
                         help='Username')
    parser.add_argument('-o', '--output-path', action='store',
                         type=str, dest='output_path', default="",
                         help='Path in which games, etc. are stored')
    parser.add_argument('--non-interactive', action='store_false',
                         dest='interactive', default=True,
                         help='Run in non-interactive mode')
    parser.add_argument('--op', action='store',
                         type=str, dest='op', default='',
                         help='What operation to run in non-interactive mode')
    parser.add_argument('--num-runs', action='store',
                         type=int, dest='num_games', default=None,
                         help='Number of runs per queuing')
    args = parser.parse_args()
    return args

class ClientOperationState:
    IDLE = 0
    RUNNING_OP = 1
    WAITING_FOR_GAME = 3
    PLAYING = 4
    PLAYING_DONE = 5
    PLAYING_QUIT = 6

class Client:

    __VERSION__ = 'ALRL2020_1.0'

    def __init__(self, 
                 username : str, 
                 password : str,
                 controller : RemoteControllerInterface, 
                 output_path : str,
                 interactive : bool = True, 
                 op : str = None,
                 num_games : Optional[int] = None,
                 server_addr : str = 'al-hockey.is.tuebingen.mpg.de',
                 server_port : str = '33000'
                ):

        self.state = ClientOperationState.IDLE

        self.interactive = interactive
        self.op = op

        self.username = username
        self.password = password
        self.controller = controller
        self.output_path = output_path

        self.client_cmd = ClientCMD(self)
        self.quering_cmd = InteractiveMode(self)

        self.verbose = True

        self.current_game = None

        self.num_games = num_games
        self.played_games = 0

        self.network_interface = NetworkInterface(
                                                  client=self,
                                                  server=server_addr,
                                                  port=server_port
                                                 )
        self.network_interface.connect()

    def stop_queuing(self) -> None:
        self.played_games = 0

        if self.state == ClientOperationState.PLAYING:
            if self.verbose:
                print('Playing last game')
            self.state = ClientOperationState.PLAYING_QUIT
        elif self.state == ClientOperationState.WAITING_FOR_GAME:
            self.waiting_for_game_loop.stop()
            del(self.waiting_for_game_loop)
            self.state = ClientOperationState.PLAYING_QUIT
            d = self.network_interface.stop_queuing()
            if not self.interactive:
                d.addCallback(self.quit)

    def _pre_issue_remote_command(self) -> None:
        self.state = ClientOperationState.RUNNING_OP

    def _post_issue_remote_command(self) -> None:
        self.state = ClientOperationState.IDLE
        self._command_line_interface()

    def _command_line_interface(self) -> None:
        if self.interactive:
            reactor.callInThread(self.client_cmd.cmdloop)
        else:
            if self.op == 'start_queuing':
                reactor.callInThread(self.quering_cmd.start_loop)
                self.start_queuing()
            else:
                raise NotImplementedError()

    def _greeting(self) -> None:
        message = '''
       Welcome to the ALRL2020 Competition.

Info:
While playing after typing start_queuing, 
you can enter the commandline interface by pressing 
escape.
'''
        return message
         
    def start_queuing(self) -> None:
        # reactor.callInThread(self.quering_cmd.stop_loop)

        if self.num_games is not None:
            if self.played_games >= self.num_games:
                self.stop_queuing()
                if not self.interactive:
                    self.quit()
                    return
                self._post_issue_remote_command()
                return

        if self.interactive:
            reactor.callInThread(self.quering_cmd.start_loop)
        self.network_interface.start_queuing()

    def quit(self, *args, **kwargs) -> None:
        self._pre_issue_remote_command()
        self.network_interface.disconnect()

    # Functions called by network_interface
    def connection_error(self, 
                         conn_err,
                        ) -> None:

        if conn_err == NetworkInterfaceConnectionError.CONNECTING:
            print('Could not connect to server. Please try again later or contact the administrator')
        elif conn_err == NetworkInterfaceConnectionError.LOST:
            print('Connection to Server lost. Please try again later or contact the administrator')

    def post_connection_established(self) -> None:
        print('Successfully connected to server')
        print(self._greeting())
        self._command_line_interface()

    # Callback functions
    def waiting_for_game_to_start(self, *args, **kwargs) -> None:
        def f():
            if self.verbose:
                print('Waiting for other player')
        if not self.state == ClientOperationState.PLAYING:
            self.state = ClientOperationState.WAITING_FOR_GAME
            self.waiting_for_game_loop = task.LoopingCall(f)
            self.waiting_for_game_loop.start(10.0)

    # Game loop functions
    def game_starts(self, 
                    ob : List[float], 
                    info : Dict) -> None:
                    
        if self.verbose:
            print(f'New Game started ({info["id"]})')
            print(f'{info["player"][0]} vs {info["player"][1]}')

        self.state = ClientOperationState.PLAYING

        if hasattr(self, 'waiting_for_game_loop'):
            self.waiting_for_game_loop.stop()
            del(self.waiting_for_game_loop)

        action = self.controller.act(np.asarray(ob)).tolist()

        self.current_game = Game(identifier=info['id'],
                                 player_one=info["player"][0],
                                 player_two=info["player"][1],
                                 fst_obs=ob,
                                 fst_action=action
                                )

        self.network_interface.send_action(action)

    def step(self, 
             ob : List[float], 
             r : Optional[int] = None, 
             done : Optional[int] = None, 
             info : Optional[Dict] = None
            ) -> None:

        action = self.controller.act(np.asarray(ob)).tolist()
        self.network_interface.send_action(action)

        self.current_game.add_transition(next_obs=ob, 
                                         next_action=action, 
                                         r=r, 
                                         done=done, 
                                         info=info
                                        )

    def game_aborted(self, 
                     msg : str
                    ) -> None:
                    
        if self.verbose:
            print(msg)

        self.current_game = None

        if self.state == ClientOperationState.PLAYING:
            self.state = ClientOperationState.PLAYING_DONE
            self.network_interface.start_queuing()
        if self.state == ClientOperationState.PLAYING_QUIT:
            self._post_issue_remote_command()

    def game_done(self, 
                  ob : List[float], 
                  r : int, 
                  done : int, 
                  info : Dict,
                  winner : str
                 ) -> None:

        if self.verbose:
            print(f'{winner} won the game')

        self.current_game.add_transition(next_obs=ob, 
                                         next_action=None, 
                                         r=r, 
                                         done=done, 
                                         info=info
                                        )
        self.current_game.save(output_path=self.output_path)
        self.current_game = None

        self.played_games += 1

        if self.state == ClientOperationState.PLAYING:
            self.state = ClientOperationState.PLAYING_DONE
            self.start_queuing()
        if self.state == ClientOperationState.PLAYING_QUIT:
            if self.interactive:
                self._post_issue_remote_command()
            else:
                self.quit()

def main(opts):
    controller = laserhockey.hockey_env.BasicOpponent(weak=False)
    client = Client(username=opts.username, 
                    controller=controller, 
                    output_path=opts.output_path,
                    interactive=opts.interactive,
                    op=opts.op,
                    num_games=opts.num_games
                   )

if __name__ == '__main__':
    opts = parseOptions()
    main(opts)