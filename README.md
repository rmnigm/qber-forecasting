# QBER Forecasting
This is a repository for ML models for quantum bit error rate (QBER) prediction.
Right now it's only research, but on real experimental data.

## Theory and problem definition

### Light Intro

There are two signal processing units - Alice and Bob.

Alice and Bob exchange quantum bit sequences and sometimes they make mistakes - optical fiber is not perfect, transmitters are not perfect, they go out of order, etc. Because of that some bits may be *flipped* and the resulting key is partly incorrect.

Here we introduce **QBER a.k.a. Quantum Bit Error Rate:**

$$
\text{QBER} = \frac{\text{number of incorrect bits}}{\text{number of received bits}}
$$

For procedure of information reconciliation we need to know QBER value before the procedure to make the IR more efficient and fast. Also, there is additional data - time series of the **channel** **intensity**:

$$
Q = \frac{\text{number of received bits}}{\text{number of sent bits}}
$$

These two time series are available for all three transmitter states - one primary and two decoy.

**Main state**
$\mu$ - main secure state, which is used for real messages. 
We do not know QBER before the correction procedure.

**Decoy states**
$\nu_{1}$ and $\nu_2$ - decoy states, which are used for calibration of the transmitters.
QBER for these states is known ahead of time and can be used in forecasting.

### Shorter than ever
We have a time series of seven variables and we need to forecast target TS value using previous data of target TS, current values of additional TS and previous values of the other TS.

## Installation
- Install Pyenv ([guide](https://github.com/pyenv/pyenv#installation))
- Install Python, used in project

  ```bash
  $ pyenv install 3.10.6
  ```
- Create virtual environment for Python and activate (will be active until you clode the terminal session or use `deactivate`):

  ```bash
  $ cd <path to cloned repo>
  $ ~/.pyenv/versions/3.10.6/bin/python -m venv qber_env
  $ source qber_env/bin/activate
  ```  
  In terminal you will now have a prefix:
  ```bash
  (qber_env)$ ...
  ```
- Check that `python` and `pip` lead to `qber_env`

    ```bash
    (qber_env)$ which python
    <path to repo>/qber_env/bin/python
    (qber_env)$ which pip
    <path to repo>/qber_env/bin/pip
    ```
- Install dependencies using requirements.txt

  ```bash
  (qber_env)$ pip install --upgrade -r requirements.txt
  ```

