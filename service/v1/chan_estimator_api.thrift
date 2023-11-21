namespace cpp chan_estimator_api

service ChanEstimatorService {
    double retrieveEst(
        1: double eMu
        2: double eMuEma
        3: double eNu1
        4: double eNu2
        5: double qMu
        6: double qNu1
        7: double qNu2
        8: bool maintenance
    )
}
