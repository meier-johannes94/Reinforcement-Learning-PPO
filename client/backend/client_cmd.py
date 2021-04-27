import sys
import cmd
import select
import tty
import termios

from twisted.internet import reactor

class InteractiveMode:

    def __init__(self, client):
        self.client = client
        self.game_loop_cmd = GameLoopCMD(client)
        self.running = False

    def start_loop(self):
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.print = print
        tty.setcbreak(sys.stdin.fileno())

        while self.running:
            self.check_for_input()

    def stop_loop(self):
        if self.running:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.running = False
            self.client.verbose = True

    def keyboard_input_available(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def check_for_input(self):
        if self.keyboard_input_available():
            esc_key_pushed = False
            while self.keyboard_input_available():  # check for key push and empty stdin (in case several keys were pushed)
                c = sys.stdin.read(1)
                if c == '\x1b':  # x1b is ESC
                    esc_key_pushed = True
            if esc_key_pushed:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                self.client.verbose = False
                self.game_loop_cmd.cmdloop()

                self.running = False
                self.client.verbose = True

class GameLoopCMD(cmd.Cmd):
    intro = 'Type help or ? to list commands.\n'
    prompt = '(cmd) '
    file = None

    def __init__(self, client):
        super().__init__()

        self.client = client

    def do_stop_queuing(self, arg):
        'Stop queuing for game'
        reactor.callFromThread(self.client.stop_queuing)
        return True

    def precmd(self, line):
        line = line.lower()
        return line

class ClientCMD(cmd.Cmd):
    intro = 'Type help or ? to list commands.\n'
    prompt = '(cmd) '
    file = None

    def __init__(self, client):
        super().__init__()

        self.client = client

    def do_start_queuing(self, arg):
        'Start queuing for game, optinal argument int: number of games to play'
        if arg != '':
            self.client.num_games = int(arg)
        reactor.callFromThread(self.client.start_queuing)
        return True

    def do_quit(self, arg):
        reactor.callFromThread(self.client.quit)
        return True

    def precmd(self, line):
        line = line.lower()
        return line