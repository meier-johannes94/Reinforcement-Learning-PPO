from typing import Optional, List, Dict

from twisted.spread import pb
from twisted.internet import reactor, defer
from twisted.cred import credentials, error as cred_error

from common.error import *

class NotConnectedError(Exception):
    pass

class NetworkInterfaceState:
    DISCONNECTED = 0
    CONNECTED = 1
    GAME_ERROR = 98
    SERVER_ERROR = 99

class NetworkInterfaceConnectionError:
    CONNECTING = 0
    LOST = 1

class NetworkInterface(pb.Referenceable):

    def __init__(self,
                 *,
                 client,
                 server : str = 'al-hockey.is.tuebingen.mpg.de', 
                 port : str = '33000'
                ):

        self.client = client

        self.server = server
        self.port = port

        self.factory = pb.PBClientFactory()
        self.connector = reactor.connectTCP(self.server, int(self.port), self.factory)

        self.remote_avatar = None
        self.remote_client = None
        self.server_version = None

        self.state = NetworkInterfaceState.DISCONNECTED

    # Functions to establish/close connection to server
    def connect(self) -> None:
        d = self.factory.login(credentials.UsernamePassword(self.client.username.encode('utf-8'), self.client.password.encode('utf-8')), client=self)
        d.addCallback(self.set_remote_avatar)
        d.addCallback(self.check_server_client_compatibility)
        d.addCallback(self.request_remote_client)
        d.addCallback(self.connected)
        d.addErrback(self.server_client_version_missmatch_error)
        d.addErrback(self.authentification_error)
        d.addErrback(self.connection_error, conn_err=NetworkInterfaceConnectionError.CONNECTING)
        reactor.run()

    def disconnect(self) -> None:
        try:
            reactor.stop()
        except:
            pass
        self.state = NetworkInterfaceState.DISCONNECTED

    def set_remote_avatar(self, 
                          avatar : pb.RemoteReference
                         ) -> None:
        self.remote_avatar = avatar

    def request_remote_client(self, *args) -> defer.Deferred:
        d = self.remote_avatar.callRemote('request_remote_client', self)
        d.addCallback(self.set_remote_client)
        
        return d

    def set_remote_client(self, 
                          remote_client : pb.RemoteReference
                         ) -> None:
        self.remote_client = remote_client

    def check_server_client_compatibility(self, *args) -> defer.Deferred:
        d = self.remote_avatar.callRemote('check_server_client_compatibility', self.client.__VERSION__)

        return d

    def connected(self, *args):
        self.state = NetworkInterfaceState.CONNECTED
        self.client.post_connection_established()

    def connection_error(self,
                         e, 
                         conn_err : NetworkInterfaceConnectionError
                        ) -> None:

        # if e is not None:
            # print(e)
        self.state = NetworkInterfaceState.SERVER_ERROR
        self.client.connection_error(conn_err)
        self.disconnect()

    def server_client_version_missmatch_error(self, e) -> None:
        e.trap(ServerClientVersionMissmatchError)
        print(e.getErrorMessage())
        self.disconnect()

    def authentification_error(self, e) -> None:
        e.trap(cred_error.UnauthorizedLogin)
        print(f'Username {self.client.username} not known.')
        self.disconnect()

    # Remote functions called by server
    def remote_game_starts(self, 
                           ob : List[float], 
                           info : Dict
                          ) -> None:

        self.client.game_starts(ob, info)

    def remote_game_aborted(self, 
                            msg : str
                           ) -> None:

        self.client.game_aborted(msg)

    def remote_game_done(self, 
                         ob : List[float], 
                         r : int, 
                         done : int, 
                         info : Dict,
                         winner : str
                        ) -> None:

        self.client.game_done(ob, r, done, info, winner)

    def remote_receive_observation(self, 
                                   ob : List[float], 
                                   r: int, 
                                   done : int, 
                                   info : Dict
                                  ) -> None:

        self.client.step(ob, r, done, info)

    # Functions called by client
    def request_stats(self) -> None:
        try:
            d = self.remote_client.callRemote('request_stats')
            d.addCallback(self.client.show_stats)

            d.addErrback(self.connection_error, conn_err=NetworkInterfaceConnectionError.LOST)
        except pb.DeadReferenceError:
            self.connection_error(None, conn_err=NetworkInterfaceConnectionError.LOST)

    def start_queuing(self) -> None:
        try:
            d = self.remote_client.callRemote('start_queuing')
            d.addCallback(self.client.waiting_for_game_to_start)
            d.addErrback(self.connection_error, conn_err=NetworkInterfaceConnectionError.LOST)
        except pb.DeadReferenceError:
            self.connection_error(None, conn_err=NetworkInterfaceConnectionError.LOST)
    
    def stop_queuing(self) -> None:
        try:
            d = self.remote_client.callRemote('stop_queuing')
            d.addErrback(self.connection_error, conn_err=NetworkInterfaceConnectionError.LOST)
            return d
        except pb.DeadReferenceError:
            self.connection_error(None, conn_err=NetworkInterfaceConnectionError.LOST)

    # Game loop function
    def send_action(self, 
                    ac : List[float]
                   ) -> None:

        try:
            d = self.remote_client.callRemote('receive_action', ac=ac)
            d.addErrback(self.connection_error, conn_err=NetworkInterfaceConnectionError.LOST)
        except pb.DeadReferenceError:
            self.connection_error(None, conn_err=NetworkInterfaceConnectionError.LOST)
