from time import sleep

from thrift.transport.TSocket import TSocket

from generated.chan_estimator_api import ChanEstimatorService
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.protocol import TBinaryProtocol

import polars as pl
from tqdm import tqdm
from sklearn.metrics import r2_score

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
    filename = 'datasets/data.csv'
    df = pl.read_csv(filename)
    transp = TBufferedTransport(TSocket('localhost', 9090))
    open_thrift_transport(transp)
    client = ChanEstimatorService.Client(TBinaryProtocol.TBinaryProtocol(transp))
    e_mu_prev = None
    predictions, true_values, ests = [], [], []
    print('Starting!')
    for i, row in tqdm(enumerate(df.rows()), total=len(df)):
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
