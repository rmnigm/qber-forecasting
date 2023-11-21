#!/usr/bin/python3

from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport.TSocket import TServerSocket, TSocket
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.server import TServer
from threading import Thread
import argparse
from generated.chan_estimator_api import ChanEstimatorService
from estimator import Estimator


class ChanEstimatorHandler:
    def __init__(self, model_path: str, config_path: str | None):
        self.estimator = Estimator(20)
        self.estimator.load_model(model_path=model_path)

    def retrieveEst(self, eMu, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        self.estimator.update(eMu, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2)
        return self.estimator.predict()


def main():
    parser = argparse.ArgumentParser(description='ML-based channel estimator')
    parser.add_argument("--sock",
                        required=True,
                        help='chan_estimator UNIX-domain socket path')
    parser.add_argument("--modelpath",
                        required=True,
                        help='cbm model filepath')
    parser.add_argument("--configpath",
                        required=False,
                        help='feature config json for model filepath')

    args = parser.parse_args()
    handler = ChanEstimatorHandler(args.modelpath, args.configpath)
    processor = ChanEstimatorService.Processor(handler)
    # transport = TServerSocket(unix_socket=args.sock)
    transport = TServerSocket('localhost', 9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    thrift_server_thread = Thread(target=server.serve, name='Thrift server')
    thrift_server_thread.start()


if __name__ == '__main__':
    main()