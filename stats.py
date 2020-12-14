import pandas as pd
import numpy as np
from helper import *


class FactorData(pd.DataFrame):
    @property
    def _constructor(self):
        return FactorData

    @property
    def _constructor_sliced(self):
        return pd.Series

    @property
    def fdate(self):
        return self._fdate

    @fdate.setter
    def fdate(self, value):
        self._fdate = value

    @property
    def fproduct(self):
        return self._fproduct

    @fproduct.setter
    def fproduct(self, value):
        self._fproduct = value

    @property
    def fHEAD_PATH(self):
        return self._fHEAD_PATH

    @fHEAD_PATH.setter
    def fHEAD_PATH(self, value):
        self._fHEAD_PATH = value

    def __getitem__(self, key):
        try:
            s = super().__getitem__(key)
        except KeyError:
            s = load(
                self._fHEAD_PATH
                + "/tmp pkl/"
                + self._fproduct
                + "/"
                + key
                + "/"
                + self._fdate
            )
            self[key] = s
        return s


import inspect
from collections import OrderedDict


class factor_template(object):
    factor_name = ""

    params = OrderedDict([("period", np.power(2, range(10, 13)))])

    def formula(self):
        pass

    def form_info(self):
        return inspect.getsource(self.formula)

    def info(self):
        info = ""
        info = info + "factor_name:\n"
        info = info + self.factor_name + "\n"
        info = info + "\n"
        info = info + "formula:\n"
        info = info + self.form_info() + "\n"
        info = info + "\n"
        info = info + "params:\n"
        for key in self.params.keys():
            info = info + "$" + key + ":" + str(self.params.get(key)) + "\n"
        return info

    def __repr__(self):
        return self.info()

    def __str__(self):
        return self.info()


def build_simple_signal(file_name, signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())

    data = load(file_name)
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))

        path = (
            HEAD_PATH
            + "/tmp pkl/"
            + product
            + "/"
            + signal_name
            + "/"
            + file_name[-12:]
        )
        S = signal_list.formula(data, *cartesian)
        save(S, path)


def build_range_signal(file_name, signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())

    data = load(file_name)
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        # signal_names.append(signal_name)
        path = (
            HEAD_PATH
            + "/tmp pkl/"
            + product
            + "/"
            + signal_name
            + "/"
            + file_name[-12:]
        )
        S = signal_list.formula(data, *cartesian)
        save(S, path)


def build_composite_signal(file_name, signal_list, product, HEAD_PATH, n=12):
    keys = list(signal_list.params.keys())
    raw_data = load(file_name)
    data = FactorData(raw_data)
    data.fdate = file_name[-n:]
    data.fproduct = product
    data.fHEAD_PATH = HEAD_PATH
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        path = (
            HEAD_PATH + "/tmp pkl/" + product + "/" + signal_name + "/" + file_name[-n:]
        )
        S = signal_list.formula(data, *cartesian)
        save(S, path)


def construct_composite_signal(
    dire_signal,
    range_signal,
    period_list,
    good_night_list,
    CORE_NUM,
    product,
    HEAD_PATH,
    min_pnl=2,
    period=4096,
    tranct=0.25e-4,
    tranct_ratio=True,
):
    from collections import OrderedDict

    class foctor_xx_period(factor_template):
        factor_name = dire_signal + "." + range_signal + ".period"
        params = OrderedDict([("period", period_list)])

        def formula(self, data, period):
            return (
                data[dire_signal + "." + str(period)]
                * data[range_signal + "." + str(period)]
            ).values

    xx = foctor_xx_period()
    create_signal_path(xx, product, HEAD_PATH)
    parLapply(
        CORE_NUM,
        good_night_list,
        build_composite_signal,
        signal_list=xx,
        product=product,
        HEAD_PATH=HEAD_PATH,
    )
    new_signal = dire_signal + "." + range_signal + "." + str(period)
    good_night_files = np.array([x[-12:] for x in good_night_list])
    all_signal = get_all_signal(good_night_files, product, new_signal, period)
    open_thre = np.quantile(
        abs(all_signal),
        np.append(np.arange(0.991, 0.999, 0.001), np.arange(0.9991, 0.9999, 0.0001)),
    )
    thre_mat = pd.DataFrame(
        data=OrderedDict([("open", open_thre), ("close", -open_thre)])
    )
    print("reverse=1")
    signal_stat = get_signal_stat(
        new_signal,
        thre_mat,
        product,
        good_night_files,
        reverse=1,
        tranct=tranct,
        tranct_ratio=tranct_ratio,
        min_pnl=min_pnl,
        CORE_NUM=CORE_NUM,
    )
    print("reverse=-1")
    signal_stat = get_signal_stat(
        new_signal,
        thre_mat,
        product,
        good_night_files,
        reverse=-1,
        tranct=tranct,
        tranct_ratio=tranct_ratio,
        min_pnl=min_pnl,
        CORE_NUM=CORE_NUM,
    )
