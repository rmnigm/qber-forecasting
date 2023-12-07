import argparse
from time import sleep

import polars as pl
from tqdm import tqdm
from sklearn.metrics import r2_score
from thrift.transport.TSocket import TSocket
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.protocol import TBinaryProtocol

from chan_estimator_api import ChanEstimatorService


THRIFT_CONNECT_RETRY_TIMEOUT = 5


def open_thrift_transport(thrift_transport):
    retries = 100
    while retries:
        try:
            thrift_transport.open()
            break
        except TTransportException:
            retries -= 1
            sleep(THRIFT_CONNECT_RETRY_TIMEOUT)
            continue
        except IOError:
            raise
    else:
        raise IOError("Could not connect to thrift server after 100 retries")


def main():
    parser = argparse.ArgumentParser(description='ML-based estimator test client')
    parser.add_argument("--sock",
                        required=True,
                        help='chan_estimator UNIX-domain socket path')
    parser.add_argument("--data",
                        required=True,
                        help='chan_estimator test dataset')
    args = parser.parse_args()
    
    transp = TBufferedTransport(TSocket(unix_socket=args.sock))
    open_thrift_transport(transp)
    client = ChanEstimatorService.Client(TBinaryProtocol.TBinaryProtocol(transp))
    
    df = pl.read_csv(args.data)
    test_df = df[int(0.75 * len(df)):]
    e_mu_prev, predictions, true_values, ests = None, [], [], []
    print('Starting!')
    for i, row in tqdm(enumerate(test_df.rows()), total=len(test_df)):
        row_id, e_mu, *values = row
        e_mu_ema, e_nu_1, e_nu_2, q_mu, q_nu1, q_nu2 = values
        prediction = client.retrieveEst(e_mu_prev, e_mu_ema, e_nu_1, e_nu_2, q_mu, q_nu1, q_nu2, False)
        predictions.append(prediction)
        true_values.append(e_mu)
        ests.append(e_mu_ema)
        e_mu_prev = e_mu
    print('R2 Score:')
    print(r2_score(true_values, predictions))


if __name__ == '__main__':
    main()
