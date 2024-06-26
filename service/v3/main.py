#!/usr/bin/python3

import argparse
from threading import Thread


from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport.TSocket import TServerSocket
from thrift.server import TServer


from estimator import Estimator
from generated.chan_estimator_api import ChanEstimatorService


class ChanEstimatorHandler:
    def __init__(self, model_path_prefix: str):
        print('Creating Estimator object...')
        self.estimator = Estimator(20)
        print('Loading model...')
        self.estimator.load_model(path_prefix=model_path_prefix)
        print('Model loaded.')

    def retrieveEst(self, eMu, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        self.estimator.update(eMu, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2)
        return self.estimator.predict()


def main():
    parser = argparse.ArgumentParser(description='ML-based channel estimator')
    parser.add_argument("--model", required=False, help='model files path prefix')
    parser.set_defaults(model='models/composite')
    
    args = parser.parse_args()
    handler = ChanEstimatorHandler(args.model)
    processor = ChanEstimatorService.Processor(handler)
    transport = TServerSocket('0.0.0.0', 8080)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    thrift_server_thread = Thread(target=server.serve, name='Thrift server')
    thrift_server_thread.start()


if __name__ == '__main__':
    main()
