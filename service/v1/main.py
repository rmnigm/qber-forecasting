#!/usr/bin/python3

import argparse
from threading import Thread


from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport.TSocket import TServerSocket, TSocket
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.server import TServer


from estimator import Estimator
from generated.chan_estimator_api import ChanEstimatorService


class ChanEstimatorHandler:
    def __init__(self, model_path: str, config_path: str | None):
        print('Creating Estimator object...')
        self.estimator = Estimator(20, feature_config_path=config_path)
        print('Loading model...')
        self.estimator.load_model(model_path=model_path)
        print('Model loaded.')

    def retrieveEst(self, eMu, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        self.estimator.update(eMu, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2)
        return self.estimator.predict()


def main():
    parser = argparse.ArgumentParser(description='ML-based channel estimator')
    parser.add_argument("--sock",
                        required=True,
                        help='chan_estimator UNIX-domain socket path')
    parser.add_argument("--model",
                        required=True,
                        help='cbm model filepath')
    parser.add_argument("--config",
                        required=False,
                        help='feature config json for model filepath')

    args = parser.parse_args()
    handler = ChanEstimatorHandler(args.model, args.config)
    processor = ChanEstimatorService.Processor(handler)
    transport = TServerSocket(unix_socket=args.sock)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    thrift_server_thread = Thread(target=server.serve, name='Thrift server')
    thrift_server_thread.start()


if __name__ == '__main__':
    main()
